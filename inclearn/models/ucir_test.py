import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, losses, network, schedulers, utils
from inclearn.models.icarl import ICarl


class UCIRTest(ICarl):
    """Implements Learning a Unified Classifier Incrementally via Rebalancing

    * http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", "icarl")
        self._n_classes = 0

        self._use_mimic_score = args.get("mimic_score", False)
        self._use_less_forget = args.get("less_forget", False)
        self._lambda_schedule = args.get("lambda_schedule", False)
        self._ranking_loss = args.get("ranking_loss", {})

        self._use_relative_teachers = args.get("relative_teachers", False)
        self._relative_teachers_old = args.get("relative_teacher_on_memory", False)

        self._gor_config = args.get("gor_config", {})

        self._use_ams_ce = args.get("adaptative_margin_softmax", False)

        self._use_attention_residual = args.get("attention_residual", False)

        self._use_teacher_confidence = args.get("teacher_confidence", False)

        self._groupwise_factors = args.get("groupwise_factors", {})

        if args.get("proxy_nca", False):
            raise Exception("Use proxy_nca_config")
        self._proxy_nca_config = args.get("proxy_nca_config", {})

        self._triplet_config = args.get("triplet_config", {})
        self._use_npair = args.get("use_npair", False)
        self._use_mer = args.get("use_mer", False)

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._weights_orthogonality = args.get("weights_orthogonality")
        self._orthoreg_config = args.get("orthoreg_config", {})
        self._dso_config = args.get("dso_config", {})
        self._mc_config = args.get("mc_config", {})
        self._srip_config = args.get("srip_config", {})

        self._rotations_config = args.get("rotations_config", {})

        classifier_kwargs = args.get("classifier_config", {})
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True,
            attention_hook=True,
            rotations_predictor=bool(self._rotations_config),
            dropout=args.get("dropout")
        )

        self._warmup_config = args.get("warmup")
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._lambda = args.get("base_lambda", 5)
        self._herding_indexes = []
        self._herding_compressed_indexes = []

        self._weight_generation = args.get("weight_generation")
        self._compressed_memory = args.get("compressed_memory")
        self._alternate_training_config = args.get("alternate_training")

        self._compressed_data = {}
        self._compressed_targets = {}
        self._compressed_means = []

        self._saved_network = None
        self._post_processing_type = None

        self._args = args
        self._args["_logs"] = {}

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._compressed_memory:
            return self._compressed_memory["quantity_images"]
        elif self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):
        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        if self._alternate_training_config and self._task != 0:
            return self._alternate_training(train_loader, val_loader)

        print("nb ", len(train_loader.dataset))
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

        self._post_processing_type = None

        if self._finetuning_config and self._finetuning_config["checkpoint"]:
            self._saved_network = self._network.copy()
        else:
            self._saved_network = None

        if self._finetuning_config:
            print("Fine-tuning")
            if self._finetuning_config["scaling"]:
                print(
                    "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
                )
                self._post_processing_type = self._finetuning_config["scaling"]

            self.build_examplars(self.inc_dataset)
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())

            self._optimizer = factory.get_optimizer(
                self._network.parameters(), self._opt_name, 0.001, self.weight_decay
            )
            self._scheduler = None
            self._training_step(
                loader, val_loader, self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"]
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _alternate_training(self, train_loader, val_loader):
        for phase in self._alternate_training_config:
            if phase["update_theta"]:
                print("Updating theta")
                for class_index in range(self._n_classes - self._task_size, self._n_classes):
                    _, loader = self.inc_dataset.get_custom_loader([class_index])
                    features, _ = utils.extract_features(self._network, loader)
                    features = F.normalize(torch.from_numpy(features), p=2, dim=1)
                    mean = torch.mean(features, dim=0)
                    mean = F.normalize(mean, dim=0, p=2)

                    self._network.classifier.weights.data[class_index] = mean.to(self._device)

            self._network.freeze(trainable=phase["train_f"], model="convnet")
            self._network.freeze(trainable=phase["train_theta"], model="classifier")
            print("Freeze convnet=" + str(phase["train_f"]))
            print("Freeze classifier=" + str(phase["train_theta"]))

            self._optimizer = factory.get_optimizer(
                self._network.parameters(), self._opt_name, self._lr, self._weight_decay
            )
            self._training_step(train_loader, val_loader, 0, phase["nb_epochs"])

    def _after_task(self, inc_dataset):
        self._monitor_scale()
        super()._after_task(inc_dataset)

        if self._compressed_memory:
            self.add_compressed_memory()

    def add_compressed_memory(self):
        _, _, self._herding_compressed_indexes, _ = self.build_examplars(
            self.inc_dataset, self._herding_compressed_indexes, self.quantity_compressed_embeddings
        )

        # Computing the embeddings of only the current task images:
        for class_index in range(self._n_classes - self._task_size, self._n_classes):
            _, loader = self.inc_dataset.get_custom_loader([class_index])
            features, targets = utils.extract_features(self._network, loader)

            selected_features = features[self._herding_compressed_indexes[class_index]]
            selected_targets = targets[self._herding_compressed_indexes[class_index]]

            self._compressed_means.append(np.mean(selected_features, axis=0))

            self._compressed_data[class_index] = selected_features
            self._compressed_targets[class_index] = selected_targets

        print(
            "{} compressed memory, or {} per class.".format(
                sum(len(x) for x in self._compressed_data.values()),
                self.quantity_compressed_embeddings
            )
        )

        # Taking in account the mean shift of the class:
        if self._compressed_memory["mean_shift"]:
            print("Computing mean shift")
            for class_index in range(self._n_classes - self._task_size):
                class_memory, class_targets = utils.select_class_samples(
                    self._data_memory, self._targets_memory, class_index
                )

                _, loader = self.inc_dataset.get_custom_loader(
                    [], memory=((class_memory, class_targets))
                )
                features, _ = utils.extract_features(self._network, loader)
                features_mean = np.mean(features, axis=0)

                diff_mean = features_mean - self._compressed_means[class_index]

                self._compressed_data[class_index] += diff_mean

        for class_index in range(self._n_classes):
            indexes = np.random.permutation(self.quantity_compressed_embeddings)
            self._compressed_data[class_index] = self._compressed_data[class_index][indexes]

    @property
    def quantity_compressed_embeddings(self):
        assert self._compressed_memory

        embed_size = 64 * 16
        image_size = 32 * 32 * 3 * 8
        total_mem = image_size * 20

        return (total_mem - image_size * self._compressed_memory["quantity_images"]) // embed_size

    def _monitor_scale(self):
        if "scale" not in self._args["_logs"]:
            self._args["_logs"]["scale"] = []

        if self._network.post_processor is None:
            s = None
        elif hasattr(self._network.post_processor, "scale"):
            s = self._network.post_processor.scale.item()
        elif hasattr(self._network.post_processor, "factor"):
            s = self._network.post_processor.factor
        print("Scale is {}.".format(s))

        self._args["_logs"]["scale"].append(s)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type == "knn":
            print("knn", self._evaluation_config)
            _, loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())
            features, targets = utils.extract_features(self._network, loader)
            features_test, targets_test = utils.extract_features(self._network, test_loader)

            return utils.apply_knn(
                features,
                targets,
                features_test,
                targets_test,
                nb_neighbors=self._evaluation_config["nb_neighbors"],
                pre_normalize=self._evaluation_config.get("pre_normalize", False)
            )
        elif self._evaluation_type == "kmeans":
            print("kmeans + knn", self._evaluation_config)
            _, loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())
            features, targets = utils.extract_features(self._network, loader)

            means, means_targets = utils.apply_kmeans(
                features,
                targets,
                nb_clusters=self._evaluation_config.get("nb_clusters", 1),
                pre_normalize=self._evaluation_config.get("pre_normalize", False)
            )
            features_test, targets_test = utils.extract_features(self._network, test_loader)

            return utils.apply_knn(
                means,
                means_targets,
                features_test,
                targets_test,
                nb_neighbors=self._evaluation_config["nb_neighbors"],
                pre_normalize=self._evaluation_config.get("pre_normalize", False)
            )
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for inputs, targets, _ in test_loader:
                ytrue.append(targets.numpy())

                inputs = inputs.to(self._device)
                logits = self._network(inputs)[1]

                preds = logits.argmax(dim=1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic", self._n_classes,
                self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        if self._saved_network is not None:
            print("Re-using saved network before finetuning")
            self._network = self._saved_network

        self._gen_weights()
        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors:
            params = []
            for group_name, group_params in self._network.group_parameters.items():
                params.append({"params": group_params,
                               "lr": self._lr * self._groupwise_factors.get(group_name, 1.0)})
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(
            params, self._opt_name, self._lr, self.weight_decay
        )

        if isinstance(self._scheduling, list):
            base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self._optimizer, self._scheduling, gamma=self._lr_decay
            )
        elif self._scheduling == "cosine":
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, self._n_epochs
            )

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                print("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

        if self._compressed_memory:
            self._compressed_iterator = 0
            self._compressed_step = self.quantity_compressed_embeddings // len(train_loader)

    def _sample_compressed(self):
        features, logits, targets = [], [], []

        low_index = self._compressed_iterator * self._compressed_step
        self._compressed_iterator += 1
        high_index = self._compressed_iterator * self._compressed_step

        for class_index in self._compressed_data.keys():
            f = self._compressed_data[class_index][low_index:high_index]
            t = self._compressed_targets[class_index][low_index:high_index]

            f = torch.tensor(f).to(self._device)
            t = torch.tensor(t).to(self._device)

            logits.append(self._network.classifier(f))
            features.append(f)
            targets.append(t)

        return torch.cat(features), torch.cat(logits), torch.cat(targets)

    def _compute_loss(self, inputs, features_logits, targets, onehot_targets, memory_flags):
        features, logits, atts = features_logits

        if self._compressed_memory and len(self._compressed_data) > 0:
            c_f, c_l, c_t = self._sample_compressed()
            features = torch.cat((features, c_f))
            logits = torch.cat((logits, c_l))
            targets = torch.cat((targets, c_t))

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type

        if self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits, old_atts = self._old_model(inputs)

            if self._compressed_memory and len(self._compressed_data) > 0:
                old_features = torch.cat((old_features, c_f))
                old_logits = torch.cat((old_logits, self._old_model.classifier(c_f)))

        if self._lambda_schedule:
            scheduled_lambda = self._lambda * math.sqrt(self._n_classes / self._task_size)
        else:
            scheduled_lambda = 1.

        if self._use_ams_ce:
            loss = losses.additive_margin_softmax_ce(logits, targets)
            self._metrics["ams"] += loss.item()
        elif self._use_npair:
            loss = losses.n_pair_loss(logits, targets)
            self._metrics["npair"] += loss.item()
        elif self._proxy_nca_config:
            if self._network.post_processor:
                self._proxy_nca_config["s"] = self._network.post_processor.factor

            loss = losses.proxy_nca_github(
                scaled_logits, targets, self._n_classes, **self._proxy_nca_config
            )
            self._metrics["nca"] += loss.item()
        elif self._triplet_config:
            loss = losses.triplet_loss(features, targets, **self._triplet_config)
            self._metrics["tri"] += loss.item()
        else:
            if self._use_teacher_confidence and self._old_model is not None:
                loss = losses.cross_entropy_teacher_confidence(
                    scaled_logits, targets, F.softmax(old_logits, dim=1), memory_flags
                )
                self._metrics["clf_conf"] += loss.item()
            else:
                loss = F.cross_entropy(scaled_logits, targets)
                self._metrics["clf"] += loss.item()

        if self._weights_orthogonality is not None:
            margin = self._weights_orthogonality.get("margin")
            ortho_loss = scheduled_lambda * losses.weights_orthogonality(
                self._network.classifier.weights, margin=margin
            )
            loss += ortho_loss
            self._metrics["ortho"] += ortho_loss.item()

        if self._gor_config:
            gor_loss = losses.global_orthogonal_regularization(
                features, targets, **self._gor_config
            )
            self._metrics["gor"] += gor_loss.item()
            loss += gor_loss

        if self._orthoreg_config:
            orthoreg_loss = losses.ortho_reg(
                self._network.classifier.weights, self._orthoreg_config
            )
            self._metrics["orthoreg"] += orthoreg_loss.item()
            loss += orthoreg_loss

        if self._dso_config:
            dso_loss = losses.double_soft_orthoreg(
                self._network.classifier.weights, self._dso_config
            )
            self._metrics["dso"] += dso_loss.item()
            loss += dso_loss

        if self._mc_config:
            mc_loss = losses.mutual_coherence_regularization(
                self._network.classifier.weights, self._mc_config
            )
            self._metrics["mc"] += mc_loss.item()
            loss += mc_loss

        if self._srip_config:
            srip_loss = losses.spectral_restricted_isometry_property_regularization(
                self._network.classifier.weights, self._srip_config
            )
            self._metrics["srip"] += srip_loss.item()
            loss += srip_loss

        if self._old_model is not None:
            if self._use_less_forget:
                distil_loss = scheduled_lambda * losses.embeddings_similarity(
                    old_features, features
                )
                loss += distil_loss
                self._metrics["dis"] += distil_loss.item()
            elif self._use_mimic_score:
                old_class_logits = logits[..., :self._n_classes - self._task_size]
                old_class_old_logits = old_logits[..., :self._n_classes - self._task_size]

                mimic_loss = F.mse_loss(old_class_logits, old_class_old_logits)
                mimic_loss *= (self._n_classes - self._task_size)
                loss += mimic_loss
                self._metrics["mimic"] += mimic_loss.item()

            if self._ranking_loss:
                ranking_loss = self._ranking_loss["factor"] * losses.ucir_ranking(
                    logits,
                    targets,
                    self._n_classes,
                    self._task_size,
                    nb_negatives=self._ranking_loss["nb_negatives"],
                    margin=self._ranking_loss["margin"]
                )
                loss += ranking_loss
                self._metrics["rank"] += ranking_loss.item()

            if self._use_relative_teachers:
                if self._relative_teachers_old:
                    indexes_old = memory_flags.eq(1.)
                    old_features_memory = old_features[indexes_old]
                    new_features_memory = features[indexes_old]
                else:
                    old_features_memory = old_features
                    new_features_memory = features

                relative_t_loss = losses.relative_teacher_distances(
                    old_features_memory, new_features_memory
                )
                loss += scheduled_lambda * relative_t_loss
                self._metrics["rel"] += relative_t_loss.item()

            if self._use_attention_residual:
                attention_loss = losses.residual_attention_distillation(old_atts, atts)
                loss += attention_loss
                self._metrics["att"] += attention_loss.item()

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss
