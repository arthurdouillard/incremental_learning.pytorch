import copy
import functools
import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from inclearn.lib import data, distance, factory, loops, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.lib.network.word import Word2vec
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class ZIL(ICarl):

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._groupwise_lr = args.get("groupwise_lr", {})

        self._finetuning_config = args.get("finetuning", {})

        self._generate_unseen = args.get("generate_unseen", {})

        # Losses definition
        self._gmm_config_loss = args.get("gmm_config", {})
        self._attention_residual_config = args.get("attention_residual", {})
        self._less_forget_config = args.get("less_forget", {})
        self._semantic_reg = args.get("semantic_regularization", {})
        self._ghost_semantic_reg = args.get("ghost_semantic_regularization", {})
        self._ghost_features = None
        self._rotation_prediction = args.get("rotation_prediction", {})

        self._gor_config = args.get("gor", {})

        self._ams_config = args.get("adaptative_margin_softmax", {})
        self._softmax_ce = args.get("softmax_ce", False)

        logger.info("Initializing ZIL")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config"),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=True,
            return_features=True,
            attention_hook=True,
            rotations_predictor=bool(self._rotation_prediction)
        )

        if args.get("word_embeddings"):
            self._word_embeddings = Word2vec(**args["word_embeddings"], device=self._device)
        else:
            self._word_embeddings = None
        self._args_we = args["word_embeddings"]

        self._n_classes = 0
        self._old_model = None

        self._data_memory, self._targets_memory = None, None
        self._examplars = {}
        self._class_means = None
        self._herding_indexes = []
        self._fixed_memory = args.get("fixed_memory", True)
        self._memory_size = args["memory_size"]
        self._herding_selection = {"type": "icarl"}

        self._all_test_classes = args["all_test_classes"]

    # ---------
    # Utilities
    # ---------

    def set_task_info(self, task_info):
        super().set_task_info(task_info)
        if self._task == 0 and self._args_we:
            self.similarity_matrix = generate_similarity_matrix(
                self._word_embeddings, list(range(self._total_n_classes))
            )

    def get_class_label(self, fake_class_ids):
        return torch.tensor([self.inc_dataset.class_order[0][i] for i in fake_class_ids])

    @staticmethod
    def get_param_groups(network, config, base_lr):
        """Returns the parameters per group with their own learning rate.

        :param network: The network whose parameters will be optimized.
        :param config: The config defining which parameters are learned, and how much.
        :param default_lr: A base learning rate
        :return: A list of dicts likewise {"params": <parameters>, "lr": <lr>}.
        """
        groups = []

        for group_name, group_parameters in network.get_group_parameters().items():
            if group_parameters is None or config.get(group_name) == 0.:
                continue

            group_lr = config.get(group_name, 1.0) * base_lr
            logger.info(f"{group_name}, lr: {group_lr}")
            groups.append({"params": group_parameters, "lr": group_lr})

        return groups

    def setup_training(self, config):
        groups = self.get_param_groups(self._network, config["groupwise_lr"], config["lr"])
        optimizer = factory.get_optimizer(
            groups, config["optimizer"], config["lr"],
            config.get("weight_decay", self._weight_decay)
        )
        scheduler = factory.get_lr_scheduler(
            config["scheduling"], optimizer, nb_epochs=config["epochs"], task=self._task
        )

        return optimizer, scheduler

    # ---------------------------
    # Crude descriptions of steps
    # Remember that resume skip the training step but do the before and after task.
    # ---------------------------

    def _before_task(self, train_loader, val_loader):
        utils.add_new_weights(
            self._network, {"type": "imprinted"} if self._task > 1 else {"type": "basic"},
            self._n_classes, self._task_size, self.inc_dataset
        )
        self._n_classes += self._task_size

        groups = self.get_param_groups(self._network, self._groupwise_lr, self._lr)
        self._optimizer = factory.get_optimizer(
            groups, self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

    def _train_task(self, train_loader, val_loader):
        if self.unsupervised_config:
            self.train_unsupervised(train_loader, val_loader)

        if self.supervised_config \
           and (self.supervised_config.get("first_task") or self._task > 0):

            if self._task > 0:
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                train_loader = self.inc_dataset.get_memory_loader(*self.get_memory())

            self.train_supervised(train_loader, val_loader)


    # --------------
    # Training steps
    # --------------

    def train_unsupervised(self, train_loader, val_loader):
        logger.info("Training ConvNet with rotations prediction.")

        optimizer, scheduler = self.setup_training(self.unsupervised_config)
        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self.unsupervised_config["epochs"],
            optimizer,
            scheduler=scheduler,
            train_function=self.forward_unsupervised,
            task=self._task,
            n_tasks=self._n_tasks,
            disable_progressbar=self._disable_progressbar
        )

    def train_supervised(self, train_loader, val_loader):
        logger.info("Finetuning ConvNet and classifier")

        optimizer, scheduler = self.setup_training(self.supervised_config)
        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self.supervised_config["epochs"],
            optimizer,
            scheduler=scheduler,
            train_function=self.forward_supervised,
            task=self._task,
            n_tasks=self._n_tasks,
            disable_progressbar=self._disable_progressbar
        )

    def train_gmmn(self):
        logger.info("Training generator GMMN")

        config = self._gmmn_config_loss

        if config.get("reinit", "always") \
           or (config.get("reinit", "first") and self._task == 0):
            logger.info("Reinit GMMN")
            self._word_embeddings = Word2vec(**self._args_we, device=self._device)
        elif config.get("reinit"):
            raise NotImplementedError(f"Unknown value for GMMN")

        optimizer = factory.get_optimizer(
            [{
                "params": self._word_embeddings.parameters(),
                "lr": config["lr"]
            }], config["optimizer"], config["lr"], config.get("weight_decay", self._weight_decay)
        )

        self._visual_features, self._visual_targets = loops.perclass_loop(
            self.inc_dataset,
            list(range(0, self._n_classes)),  # All seen classes
            self._multiple_devices,
            config["epochs"],
            optimizer,
            self.forward_gmmn,
            self._task,
            self._n_tasks,
            network=self._network,
            word_embeddings=self._word_embeddings,
            target_to_word=self.get_class_label,
            disable_progressbar=self._disable_progressbar,
            scheduler=factory.get_lr_scheduler(
                config.get("scheduling"), optimizer, nb_epochs=config["epochs"]
            ),
            batch_size=config.get("batch_size", 128)
        )

    def train_fake_classifier(self):
        logger.info("Finetuning ConvNet and classifier")
        config = self.fakeclassifier_config

        optimizer, scheduler = self.setup_training(config)

        if config.get("online", False):
            logger.info("Finetuning fake classifier with online generation.")
            self._word_embeddings.eval()
            loops.online_generation(
                self._visual_features,
                self._visual_targets,
                config["epochs"],
                optimizer,
                self._network.classifier,
                self.forward_fakeclassifier,
                self._word_embeddings,
                self.get_class_label,
                scheduler=scheduler,
                multiply=config.get("multiply", 1)
            )
            self._word_embeddings.train()
        else:
            logger.info("Finetuning fake classifier with offline generation.")

            self._word_embeddings.eval()
            if "look_ahead" in config:
                max_class = config.get("look_ahead") + self._n_classes
            else:
                max_class = self._total_n_classes

            fake_features, fake_targets = [], []
            for class_id in range(self._n_classes, max_class):
                targets = [class_id for _ in range(config["nb_samples"])]
                cifar_targets = self.get_class_label(targets).to(self._device)
                with torch.no_grad():
                    fake_features.append(self._word_embeddings(cifar_targets))
                fake_targets.append(torch.tensor(targets).long().to(self._device))
            self._word_embeddings.train()
            features = torch.cat([self._visual_features, *fake_features])
            targets = torch.cat([self._visual_targets.to(self._device), *fake_targets])

            loops.features_to_classifier_loop(
                features,
                targets,
                config["epochs"],
                optimizer,
                self._network.classifier,
                self.forward_fakeclassifier,
                scheduler=scheduler
            )

    # -----------------
    # Losses definition
    # -----------------
    def forward_unsupervised(
        self, training_network, inputs, targets, memory_flags, metrics, **kwargs
    ):
        loss, outputs = losses.unsupervised_rotations(inputs, memory_flags, training_network)
        metrics["rot"] += loss.item()

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)

            if self._attention_residual_config:
                if self._attention_residual_config.get("scheduled_factor", False):
                    factor = self._attention_residual_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._attention_residual_config.get("factor", 1.)

                attention_loss = factor * losses.residual_attention_distillation(
                    old_outputs["attentions"],
                    outputs["attentions"],
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._attention_residual_config
                )
                loss += attention_loss
                self._metrics["att"] += attention_loss.item()

        return loss

    def forward_supervised(
        self, training_network, inputs, targets, memory_flags, metrics, **kwargs
    ):
        outputs = training_network(inputs)

        if self._ams_config:
            ams_config = copy.deepcopy(self._ams_config)
            if self._network.post_processor:
                ams_config["scale"] = self._network.post_processor.factor

            loss = losses.additive_margin_softmax_ce(outputs["logits"], targets, **ams_config)
            metrics["ams"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(self._network.post_process(outputs["logits"]), targets)
            metrics["cce"] += loss.item()
        else:
            raise ValueError("No classification loss defined!")

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)

            if self._attention_residual_config:
                if self._attention_residual_config.get("scheduled_factor", False):
                    factor = self._attention_residual_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._attention_residual_config.get("factor", 1.)

                attention_loss = factor * losses.residual_attention_distillation(
                    old_outputs["attentions"],
                    outputs["attentions"],
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._attention_residual_config
                )
                loss += attention_loss
                self._metrics["att"] += attention_loss.item()

    def forward_gmmn(self, visual_features, semantic_features):
        return mmd(visual_features, semantic_features, **self._gmm_config_loss)

    def forward_fakeclassifier(self, logits, targets):
        if self._ams_config:
            ams_config = copy.deepcopy(self._ams_config)
            if self._network.post_processor:
                ams_config["scale"] = self._network.post_processor.factor

            loss = losses.additive_margin_softmax_ce(logits, targets, **ams_config)
        elif self._softmax_ce:
            loss = F.cross_entropy(self._network.post_process(logits), targets)
        else:
            raise ValueError("No classification loss defined!")

        return loss

    # ----------------

    def _gen_ghost_features_gmm(self):
        config = self._gmm_config_loss["ghost_features"]
        classes_to_create = list(range(self._n_classes, self._n_classes + config["nb_classes"]))

        weights = []
        targets = []
        for class_id in classes_to_create:
            class_id = np.ones(config["amount_per_class"], dtype="int") * class_id
            class_id = self.get_class_label(class_id).to(self._device)

            with torch.no_grad():
                weights.append(self._word_embeddings(class_id))

            targets.extend(class_id)

        weights = torch.cat(weights).to(self._device)
        targets = torch.tensor(targets).long().to(self._device)

        return weights, targets

    def _after_task(self, inc_dataset):
        if self._gmm_config_loss:
            if not self._gmm_config_loss.get("interpolation", False):
                self._train_gmm()

            if self._gmm_config_loss["ghost_features"]:
                if self._task < self._n_tasks - 1:
                    if self._gmm_config_loss.get("interpolation", False):
                        self._ghost_features = self._gen_ghost_features_interpolation()
                        logger.info("Generating ghost features with interpolation...")
                    else:
                        logger.info("Generating ghost features with GMM...")
                        self._ghost_features = self._gen_ghost_features_gmm()
                else:
                    logger.info("Ghost features are no longer needed for last task.")
                    self._ghost_features = None

        self._old_model = self._network.copy().eval().to(self._device)
        self._network.on_task_end()

    def _gmm_loss(self, visual_features, semantic_features):
        return mmd(visual_features, semantic_features, **self._gmm_config_loss)

    def _train_gmm(self):
        logger.info("Training GMM.")

        if self._gmm_config_loss.get("reinit"):
            logger.info("Reinit GMM")
            self._word_embeddings = Word2vec(**self._args_we, device=self._device)

        optimizer = factory.get_optimizer(
            [
                {
                    "params": filter(lambda p: p.requires_grad, self._word_embeddings.parameters()),
                    "lr": self._gmm_config_loss["lr"]
                }
            ], self._gmm_config_loss["optimizer"], self._gmm_config_loss["lr"], self._weight_decay
        )

        print("Before, norm of emb ", self._word_embeddings.emb.weight.norm())
        self._visual_features, self._visual_targets = loops.perclass_loop(
            self.inc_dataset,
            list(range(0, self._n_classes)),
            self._multiple_devices,
            self._gmm_config_loss["epochs"],
            optimizer,
            self._gmm_loss,
            self._task,
            self._n_tasks,
            network=self._network,
            word_embeddings=self._word_embeddings,
            target_to_word=self.get_class_label,
            disable_progressbar=self._disable_progressbar,
            scheduler=factory.get_lr_scheduler(
                self._gmm_config_loss.get("scheduling"),
                optimizer,
                nb_epochs=self._gmm_config_loss["epochs"],
                lr_decay=self._gmm_config_loss.get("lr_decay")
            ),
            batch_size=self._gmm_config_loss.get("batch_size", 128)
        )
        print("After, norm of emb ", self._word_embeddings.emb.weight.norm())

    def _eval_task(self, loader):
        self.eval()

        ypred, ytrue = [], []

        if self._generate_unseen:
            logger.info("Generating weights for unseen classes.")
            real_clf_weights = copy.deepcopy(self._network.classifier._weights)

            if self._generate_unseen["what"] == "new":
                self._network.classifier._weights.append(
                    nn.Parameter(
                        self.get_fake_weights(
                            self._network.classifier.weights, **self._generate_unseen
                        )
                    )
                )
            elif self._generate_unseen["what"] == "all":
                if self._all_test_classes is True:
                    nb_unseen_classes = self._total_n_classes - self._n_classes
                else:
                    nb_unseen_classes = self._all_test_classes * 10

                self._network.classifier._weights = nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(self._n_classes, self._network.convnet.out_dim)),
                        nn.Parameter(torch.randn(nb_unseen_classes, self._network.convnet.out_dim))
                    ]
                )
                self._network.classifier.to(self._device)
                nn.init.kaiming_normal_(self._network.classifier._weights[0], nonlinearity="linear")
                nn.init.kaiming_normal_(self._network.classifier._weights[1], nonlinearity="linear")
            else:
                raise ValueError(self._generate_unseen["what"])

            if "training" in self._generate_unseen:
                for config in self._generate_unseen["training"]:
                    self.finetune_fake_classifier(config)

            if self._generate_unseen.get("align_weights", False):
                self._network.classifier.align_weights()
            elif self._generate_unseen.get("align_inv_weights", False):
                self._network.classifier.align_inv_weights()

        logger.info("Evaluating model...")
        for input_dict in loader:
            with torch.no_grad():
                logits = self._network(input_dict["inputs"].to(self._device))["logits"]

            ytrue.append(input_dict["targets"].numpy())
            ypred.append(logits.cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        if self._generate_unseen:
            self._network.classifier._weights = real_clf_weights

        self.train()
        return ypred, ytrue

    def finetune_fake_classifier(self, config):
        logger.info("Finetune fake classifier")

        parameters = []
        for group_name, group_params in self._network.get_group_parameters().items():
            if group_name in config["parameters"]:
                parameters.append(
                    {
                        "params": group_params,
                        "lr": config["lr"] * config["parameters"][group_name]
                    }
                )

        optimizer = factory.get_optimizer(
            parameters, config["optimizer"], config["lr"], config["lr"]
        )

        def loss_function(logits, targets):
            ams_config = copy.deepcopy(self._ams_config)
            if self._network.post_processor:
                ams_config["scale"] = self._network.post_processor.factor
            return losses.additive_margin_softmax_ce(logits, targets, **ams_config)

        self._word_embeddings.eval()
        fake_features = []
        fake_targets = []

        if self._all_test_classes is True:
            max_class = self._total_n_classes
        else:
            max_class = self._all_test_classes * 10 + self._n_classes

        scheduler = factory.get_lr_scheduler(
            config.get("scheduling"),
            optimizer,
            nb_epochs=config["epochs"],
            lr_decay=config.get("lr_decay")
        )
        if config.get("online", False):
            logger.info("Finetuning fake classifier with online generation.")
            self._word_embeddings.eval()
            loops.online_generation(
                self._visual_features,
                self._visual_targets,
                config["epochs"],
                optimizer,
                self._network.classifier,
                loss_function,
                self._word_embeddings,
                self.get_class_label,
                scheduler=scheduler,
                multiply=config.get("multiply", 1)
            )
            self._word_embeddings.train()
        else:
            logger.info("Finetuning fake classifier with offline generation.")

            for class_id in range(self._n_classes, max_class):
                targets = [class_id for _ in range(config["nb_samples"])]
                cifar_targets = self.get_class_label(targets).to(self._device)
                with torch.no_grad():
                    fake_features.append(self._word_embeddings(cifar_targets))
                fake_targets.append(torch.tensor(targets).long().to(self._device))
            self._word_embeddings.train()
            features = torch.cat([self._visual_features, *fake_features])
            targets = torch.cat([self._visual_targets.to(self._device), *fake_targets])

            loops.features_to_classifier_loop(
                features,
                targets,
                config["epochs"],
                optimizer,
                self._network.classifier,
                loss_function,
                scheduler=scheduler
            )

    def get_fake_weights(
        self, real_weights, nb_samples=100, method="word_embeddings", weight_norm=True, **kwargs
    ):
        classes_to_create = list(range(self._n_classes, self._total_n_classes))
        if method == "word_embeddings":
            self._word_embeddings.eval()
            weights = []
            for class_id in classes_to_create:
                class_id = [class_id for _ in range(nb_samples)]
                real_class_id = self.get_class_label(class_id).to(self._device)

                weights.append(self._word_embeddings(real_class_id).mean(dim=0, keepdims=True))
            weights = torch.cat(weights, dim=0)
            self._word_embeddings.train()
        elif method == "random":
            weights = torch.randn(len(classes_to_create), 128).float().to(self._device)
        else:
            raise NotImplementedError(
                "Unknown method {} to generate unseen weights.".format(method)
            )

        if weight_norm:
            avg_weights_norm = torch.mean(real_weights.data.norm(dim=1, keepdim=True))
            weights.mul_(avg_weights_norm)

        return weights

    def _accuracy(self, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(axis=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _forward_loss(self, training_network, inputs, targets, memory_flags, metrics, **kwargs):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)

        loss = self._compute_loss(
            inputs, outputs, targets, onehot_targets, memory_flags, metrics, training_network
        )

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def _forward_unsupervised_loss(
        self, training_network, inputs, targets, memory_flags, metrics, **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        rot = losses.unsupervised_rotations(
            inputs, memory_flags, training_network, **self._rotation_prediction
        )
        loss = rot
        metrics["rot"] += rot.item()

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def _compute_loss(
        self, inputs, outputs, targets, onehot_targets, memory_flags, metrics, training_network
    ):
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]
        scaled_logits = self._network.post_process(logits)

        if self._ams_config:
            ams_config = copy.deepcopy(self._ams_config)
            if self._network.post_processor:
                ams_config["scale"] = self._network.post_processor.factor

            loss = losses.additive_margin_softmax_ce(logits, targets, **ams_config)
            metrics["ams"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(scaled_logits, targets)
            metrics["cce"] += loss.item()

        if self._semantic_reg:
            sem_reg = semantic_regularization(
                features,
                self.get_class_label(targets).to(self._device), self.similarity_matrix,
                **self._semantic_reg
            )
        if self._ghost_semantic_reg and self._ghost_features is not None:
            ghost_sem_reg = ghost_semantic_regularization(
                features=features,
                targets=self.get_class_label(targets).to(self._device),
                similarity_matrix=self.similarity_matrix,
                ghost_features=self._ghost_features[0],
                ghost_targets=self._ghost_features[1],
                **self._ghost_semantic_reg
            )
            metrics["gsem"] += ghost_sem_reg.item()
            loss += ghost_sem_reg

        if self._gor_config:
            gor = losses.global_orthogonal_regularization(features, targets, **self._gor_config)
            metrics["gor"] += gor.item()
            loss += gor

        if self._old_model is not None:
            # Computing outputs using previous task's model:
            self._old_model.zero_grad()
            old_outputs = self._old_model(inputs)
            old_atts = [a.detach() for a in old_outputs["attention"]]
            old_features = old_outputs["raw_features"].detach()

            if self._less_forget_config:
                if self._less_forget_config["scheduled_factor"]:
                    factor = self._less_forget_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._less_forget_config.get("factor", 1.)

                distil_loss = factor * losses.embeddings_similarity(old_features, features)
                loss += distil_loss
                metrics["lf"] += distil_loss.item()

            if self._attention_residual_config:
                if self._attention_residual_config.get("scheduled_factor", False):
                    factor = self._attention_residual_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._attention_residual_config.get("factor", 1.)

                attention_loss = factor * losses.residual_attention_distillation(
                    old_atts,
                    atts,
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._attention_residual_config
                )
                loss += attention_loss
                metrics["att"] += attention_loss.item()

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        return loss


def semantic_regularization(
    features,
    targets,
    similarity_matrix,
    margin=None,
    aggreg="mean",
    factor=1.0,
):
    pair_indexes = []

    np_targets = targets.cpu().numpy()

    for index, target in enumerate(np_targets):
        neg_indexes = np.where(np_targets != target)[0]
        neg_index = np.random.choice(neg_indexes)
        pair_indexes.append(tuple(sorted((index, neg_index))))

    pair_indexes_ = list(set(pair_indexes))
    pair_indexes = torch.tensor(pair_indexes_).long()

    left = features[pair_indexes[..., 0]]
    right = features[pair_indexes[..., 1]]

    similarities = F.cosine_similarity(left, right)

    if margin is not None:
        margins = torch.ones_like(similarity_matrix) * margin
    else:
        margins = similarity_matrix[targets[pair_indexes[..., 0]], targets[pair_indexes[..., 1]]]

    hinges = torch.clamp(similarities - margins, min=0.)

    return factor * _aggreg(hinges, aggreg, features_dim=features.shape[1])


def ghost_semantic_regularization(
    features,
    targets,
    ghost_features,
    ghost_targets,
    similarity_matrix,
    margin=None,
    aggreg="mean",
    factor=1.0,
):
    neg_indexes = []

    np_targets = targets.cpu().numpy()
    for index, target in enumerate(np_targets):
        neg_index = np.random.choice(len(ghost_targets))
        neg_indexes.append(neg_index)

    selected_ghosts_features = ghost_features[neg_indexes]
    selected_ghosts_targets = ghost_targets[neg_indexes]

    similarities = F.cosine_similarity(features, selected_ghosts_features)

    if margin is not None:
        margins = torch.ones_like(similarity_matrix) * margin
    else:
        margins = similarity_matrix[targets, selected_ghosts_targets]

    hinges = torch.clamp(similarities - margins, min=0.)

    return factor * _aggreg(hinges, aggreg, features_dim=features.shape[1])


def _aggreg(hinges, aggreg_method, features_dim):
    if isinstance(aggreg_method, int):
        return torch.mean(torch.topk(hinges, k=aggreg_method)[0])
    elif aggreg_method == "mean":
        return torch.mean(hinges)
    elif aggreg_method == "adamine":
        nb_not_null = (torch.clamp(hinges - 1e-6, min=0.) != 0.).sum()
        if nb_not_null == 0.:
            nb_not_null = 1.
        return torch.sum(hinges) / nb_not_null
    elif aggreg_method == "gor":
        first_moment = torch.mean(hinges)
        second_moment = torch.mean(torch.pow(hinges, 2))

        return torch.pow(first_moment, 2) + torch.clamp(second_moment - 1. / features_dim, min=0.)

    raise NotImplementedError("Unknown aggreg {}.".format(aggreg_method))


def generate_similarity_matrix(word_embeddings, class_ids):
    classes = torch.tensor(class_ids).long().to(word_embeddings.device)

    with torch.no_grad():
        embeddings = word_embeddings.forward(classes, only_word=True)

    return torch.mm(embeddings, embeddings.t())


def mmd(
    x, y, sigmas=[2, 5, 10, 20, 40, 80], normalize=True, scale_matrix=False, bucher=False, **kwargs
):
    """Maximum Mean Discrepancy with several Gaussian kernels."""
    if bucher:
        return moment_loss(y, x, sigma=sigmas, device=x.device)

    if normalize:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

    xy = torch.cat((x, y), dim=0)

    if scale_matrix:
        scale = get_scale_matrix(x.shape[0], y.shape[0], x.device)
        scale = torch.matmul(scale, scale.t())
    else:
        scale = 1.

    xx = torch.mm(xy, xy.t())
    x2 = torch.sum(xx**2, dim=1, keepdim=True)

    exponent = xx - 0.5 * x2 - 0.5 * x2.t()

    loss = 0.
    for sigma in sigmas:
        kernel_val = torch.exp(1 / sigma * exponent)
        loss += torch.sum(scale * kernel_val)
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

    return torch.sqrt(loss)


def get_scale_matrix(M, N, device):
    s1 = torch.ones((N, 1)) * 1.0 / N
    s2 = torch.ones((M, 1)) * -1.0 / M
    s1, s2 = s1.to(device), s2.to(device)
    return torch.cat((s1, s2), 0)


def moment_loss(gen_samples, x, sigma, device):
    X = torch.cat((gen_samples, x), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()
    M = gen_samples.size()[0]
    N = x.size()[0]
    s = get_scale_matrix(M, N, device)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(exp / v)
        loss += torch.sum(S * kernel_val)

    loss = torch.sqrt(loss)
    return loss


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_sigmas(sigmas, device):
    sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
    return -1 / (2 * sigmas)


@functools.lru_cache(maxsize=1, typed=False)
def get_scale_matrix(M, N, device):
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    return torch.cat((s1, s2), 0).to(device)
