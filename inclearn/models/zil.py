import collections
import copy
import functools
import logging
import math
import os
import pickle

import numpy as np
import torch
from sklearn import preprocessing as skpreprocessing
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.nn import functional as F

from inclearn.lib import data, distance, factory, loops, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.lib.network.autoencoder import AdvAutoEncoder
from inclearn.lib.network.classifiers import (BinaryCosineClassifier,
                                              DomainClassifier)
from inclearn.lib.network.word import Word2vec
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class ZIL(ICarl):

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self.device = args["device"][0]
        self._multiple_devices = args["device"]

        self._weight_decay = args["weight_decay"]

        # Steps definition
        self.unsupervised_config = args.get("unsupervised", {})
        self.supervised_config = args.get("supervised", {})
        self.gmmn_config = args.get("gmmn", {})
        self.autoencoder_config = args.get("autoencoder", {})
        self.fakeclassifier_config = args.get("fake_classifier", {})

        # Losses definition
        self._pod_spatial_config = args.get("pod_spatial", {})
        self._pod_flat_config = args.get("pod_flat", {})
        self.ghost_config = args.get("ghost_regularization", {})
        self.real_config = args.get("semantic_regularization", {})
        self.hyperplan_config = args.get("hyperplan_regularization", {})
        self.placement_config = args.get("ghost_placement_config", {})
        self.adv_placement_config = args.get("adv_ghost_placement_config", {})
        self.ucir_ranking_config = args.get("ucir_ranking", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        logger.info("Initializing ZIL")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config"),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            return_features=True,
            attention_hook=True,
            rotations_predictor=True
        )
        self.args = args

        self._new_weights_config = args.get("weight_generation", {"type": "imprinted"})

        self._w2v_path = args.get("word2vec_path") or args.get("data_path")
        if args.get("word_embeddings"):
            self._word_embeddings = Word2vec(
                **args["word_embeddings"], data_path=self._w2v_path, device=self._device
            )
        else:
            self._word_embeddings = None
        self._old_word_embeddings = None

        self._args_we = args.get("word_embeddings")
        self._args_ae = args.get("autoencoder_archi")

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

        self._preprocessing = None
        self._gen_preprocessing = None
        self._svm = None

        self._ghosts = None
        self._old_ghosts = None

        self._saved_weights = None

        self._cheat_pixels = None

    # ---------
    # Utilities
    # ---------

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    def save_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        logger.info(f"Saving model at {path}.")
        torch.save(self.network.state_dict(), path)

        if self._word_embeddings is not None:
            path = os.path.join(directory, f"gen_{run_id}_task_{self._task}.pth")
            logger.info(f"Saving generator at {path}.")
            torch.save(self._word_embeddings.state_dict(), path)

    def load_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            return

        logger.info(f"Loading model at {path}.")
        try:
            self.network.load_state_dict(torch.load(path, map_location=self._device), strict=False)
        except Exception as e:
            logger.warning(f"Old method to save weights, it's deprecated!: {e}")
            self._network = torch.load(path)
        self.network.to(self._device)

        if self._saved_weights is not None:
            logger.info("Keeping previous weights anyway")
            self._network.classifier._weights = self._saved_weights

        path = os.path.join(directory, f"gen_{run_id}_task_{self._task}.pth")
        if os.path.exists(path):
            logger.info(f"Loading generator at {path}.")
            try:
                self._word_embeddings.load_state_dict(
                    torch.load(path, map_location=self._device), strict=False
                )
                self._word_embeddings.to(self._device)
            except Exception:
                logger.warning("Failed to reload generator, it was probably changed.")

    def get_class_label(self, fake_class_ids):
        """Class id in-code ==> Class id in datasets."""
        return torch.tensor([self.inc_dataset.class_order[0][i] for i in fake_class_ids])

    def get_inv_class_label(self, true_class_ids):
        """Class id in datasets ==> Class id in-code."""
        return torch.tensor([self.inc_dataset.class_order[0].index(i) for i in true_class_ids])

    @staticmethod
    def get_param_groups(network, config, base_lr, additional_parameters=None):
        """Returns the parameters per group with their own learning rate.

        :param network: The network whose parameters will be optimized.
        :param config: The config defining which parameters are learned, and how much.
        :param default_lr: A base learning rate
        :return: A list of dicts likewise {"params": <parameters>, "lr": <lr>}.
        """
        groups = []

        parameters_dict = [network.get_group_parameters()]
        if additional_parameters is not None:
            parameters_dict.append(additional_parameters)

        for group_dict in parameters_dict:
            for group_name, group_parameters in group_dict.items():
                if group_parameters is None or group_name not in config:
                    continue

                group_lr = config.get(group_name, 1.0) * base_lr
                logger.info(f"{group_name}, lr: {group_lr}")
                groups.append({"params": group_parameters, "lr": group_lr})

        return groups

    def setup_training(self, config, additional_parameters=None):
        groups = self.get_param_groups(
            self._network,
            config["groupwise_lr"],
            config["lr"],
            additional_parameters=additional_parameters
        )
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
        self._cheat_pixels = None

        if self._task > 0 and self._task != self._n_tasks - 1:
            logger.info("Setup of the constraints...")
            self._n_classes += self._task_size
            self._old_ghosts = copy.deepcopy(self._ghosts)
            self._setup_constraints()
            self._n_classes -= self._task_size

        if self._task > 1 and self._new_weights_config["type"] == "ghosts":
            self._new_weights_config["ghosts"] = self._old_ghosts

        if self._task == 0:
            utils.add_new_weights(
                self._network, {"type": "basic"}, self._n_classes, self._task_size, self.inc_dataset
            )
        elif self._task == 1:
            utils.add_new_weights(
                self._network, {"type": "imprinted"}
                if self._network.classifier.classifier_type == "cosine" else {"type": "basic"},
                self._n_classes, self._task_size, self.inc_dataset
            )
        elif self._new_weights_config["type"] == "neg_weights":
            # Take the neg weights
            logger.info("Promoting Ghost Centroids to fist-class status.")
            neg_weights = self._network.classifier._negative_weights
            to_promote = neg_weights[:self._task_size].data
            self._network.classifier.add_custom_weights(
                to_promote, ponderate=self._new_weights_config.get("ponderate")
            )
        else:  # Take mean of ghost per class
            utils.add_new_weights(
                self._network, self._new_weights_config, self._n_classes, self._task_size,
                self.inc_dataset
            )

        if self._task == self._n_tasks - 1:
            # If we are on last task, disable negative weights
            self._network.classifier._negative_weights = None

        self._n_classes += self._task_size

        if "ghosts" in self._new_weights_config:
            del self._new_weights_config["ghosts"]

    def _train_task(self, train_loader, val_loader):
        if self._cheat_pixels is not None:
            train_loader.dataset.x = np.concatenate((train_loader.dataset.x, self._cheat_pixels[0]))
            train_loader.dataset.y = np.concatenate((train_loader.dataset.y, self._cheat_pixels[1]))
            train_loader.dataset.memory_flags = np.concatenate(
                (train_loader.dataset.memory_flags, self._cheat_pixels[2])
            )

        if self.placement_config.get("initial_centroids"):
            _, loader = self.inc_dataset.get_custom_loader(
                list(range(self._n_classes - self._task_size, self._n_classes))
            )
            c_features, c_targets = utils.compute_centroids(self._network, loader)
            self._c_features = torch.tensor(c_features).float().to(self._device)
            self._c_targets = torch.tensor(c_targets).long().to(self._device)
        else:
            self._c_features, self._c_targets = None, None

        if self._task > 0 and self.adv_placement_config:
            self._network.create_domain_classifier()

        if self.unsupervised_config \
           and ((self._task > 0 and not self.unsupervised_config.get("only_first", False)) \
                or self._task == 0):
            self.train_unsupervised(train_loader, val_loader)

        if self.supervised_config:
            self.train_supervised(train_loader, val_loader)

    def _after_task(self, inc_dataset):
        if self._task != self._n_tasks - 1:
            if self.gmmn_config:
                self.train_gmmn()
            elif self.autoencoder_config:
                self.train_autoencoder()

        if self._task > 0 and self.adv_placement_config:
            self._network.del_domain_classifier()

        self._old_model = self._network.copy().eval().to(self._device)
        self._network.on_task_end()

        if self._word_embeddings is not None:
            self._old_word_embeddings = copy.deepcopy(self._word_embeddings)
            self._old_word_embeddings.eval().to(self._device)

    def _eval_task(self, loader):
        self.eval()

        ypred, ytrue = [], []

        if self.fakeclassifier_config and self._task != self._n_tasks - 1:
            logger.info("Generating weights for unseen classes.")
            real_clf_weights = copy.deepcopy(self._network.classifier._weights)
            nb_unseen_classes = self._total_n_classes - self._n_classes

            if self.fakeclassifier_config.get("what_post"):
                postprocessor = copy.deepcopy(self._network.post_processor)
                if isinstance(self.fakeclassifier_config.get("what_post"), float):
                    self._network.post_processor.factor.data.fill_(
                        self.fakeclassifier_config.get("what_post", 1.0)
                    )

            if hasattr(
                self._network.classifier, "_bias"
            ) and self._network.classifier._bias is not None:
                real_clf_bias = copy.deepcopy(self._network.classifier._bias)

            if self.fakeclassifier_config["what"] == "new":
                self._network.classifier._weights.append(
                    nn.Parameter(
                        torch.randn(
                            nb_unseen_classes * self._network.classifier.proxy_per_class,
                            self._network.convnet.out_dim
                        )
                    )
                )
                if hasattr(
                    self._network.classifier, "_bias"
                ) and self._network.classifier._bias is not None:
                    logging.info("Generating also bias.")
                    self._network.classifier._bias.append(
                        nn.Parameter(torch.zeros(nb_unseen_classes))
                    )
            elif self.fakeclassifier_config["what"] == "new_scaleOld":
                self._network.classifier._weights = nn.ParameterList(
                    [
                        nn.Parameter(
                            self._preprocessing.transform(self._network.classifier.weights.data)
                        ),
                        nn.Parameter(
                            torch.randn(
                                nb_unseen_classes * self._network.classifier.proxy_per_class,
                                self._network.convnet.out_dim
                            )
                        )
                    ]
                )
            elif self.fakeclassifier_config["what"] == "negative":
                params = [
                    nn.Parameter(
                        self._preprocessing.transform(self._network.classifier.weights.data)
                    )
                ]

                if self._task == 0:
                    params.append(
                        nn.Parameter(
                            torch.randn(
                                nb_unseen_classes * self._network.classifier.proxy_per_class,
                                self._network.convnet.out_dim
                            )
                        )
                    )
                elif isinstance(self._network.classifier._negative_weights, nn.Parameter):
                    params.append(
                        nn.Parameter(
                            self._preprocessing.transform(
                                self._network.classifier._negative_weights.data
                            )
                        )
                    )
                else:
                    params.append(
                        nn.Parameter(
                            self._preprocessing.transform(
                                self._network.classifier._negative_weights
                            )
                        )
                    )

                self._network.classifier._weights = nn.ParameterList(params)
            elif self.fakeclassifier_config["what"] == "negative_no_tsrf":
                params = [nn.Parameter(self._network.classifier.weights.data)]

                if self._task == 0:
                    params.append(
                        nn.Parameter(torch.randn(nb_unseen_classes, self._network.convnet.out_dim))
                    )
                elif isinstance(self._network.classifier._negative_weights, nn.Parameter):
                    params.append(nn.Parameter(self._network.classifier._negative_weights.data))
                else:
                    params.append(nn.Parameter(self._network.classifier._negative_weights))
                self._network.classifier._weights = nn.ParameterList(params)
            elif self.fakeclassifier_config["what"] == "negative_scaled":
                self._network.classifier._weights = nn.ParameterList(
                    [
                        nn.Parameter(
                            self._preprocessing.transform(self._network.classifier.weights.data)
                        ),
                        nn.Parameter(
                            self._preprocessing.transform(
                                self._network.classifier._negative_weights
                            )
                        )
                    ]
                )
            elif self.fakeclassifier_config["what"] == "all":
                self._network.classifier._weights = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.randn(
                                self._n_classes * self._network.classifier.proxy_per_class,
                                self._network.convnet.out_dim
                            )
                        ),
                        nn.Parameter(
                            torch.randn(
                                nb_unseen_classes * self._network.classifier.proxy_per_class,
                                self._network.convnet.out_dim
                            )
                        )
                    ]
                )
                if hasattr(
                    self._network.classifier, "_bias"
                ) and self._network.classifier._bias is not None:
                    logging.info("Generating also bias.")
                    self._network.classifier._bias = nn.ParameterList(
                        [
                            nn.Parameter(torch.randn(self._n_classes)),
                            nn.Parameter(torch.randn(nb_unseen_classes))
                        ]
                    )
                nn.init.kaiming_normal_(self._network.classifier._weights[0], nonlinearity="linear")
                nn.init.kaiming_normal_(self._network.classifier._weights[1], nonlinearity="linear")
            else:
                raise ValueError(self.fakeclassifier_config["what"])
            self._network.classifier.to(self._device)

            if "training" in self.fakeclassifier_config:
                if self.fakeclassifier_config.get("only_first_task") and self._task == 0:
                    self.train_fake_classifier(self.fakeclassifier_config["training"])
                    has_trained = True
                elif not self.fakeclassifier_config.get("only_first_task", False):
                    self.train_fake_classifier(self.fakeclassifier_config["training"])
                    has_trained = True
                else:
                    has_trained = False
            else:
                has_trained = False

            if self.fakeclassifier_config.get("postprocessing", "none") == "align_weights":
                self._network.classifier.align_weights()
            elif self.fakeclassifier_config.get("postprocessing", "none") == "align_inv_weights":
                self._network.classifier.align_inv_weights()
            elif self.fakeclassifier_config.get(
                "postprocessing", "none"
            ) == "align_inv_weights_unseen":
                logger.info("Align unseen to seen.")
                self._network.classifier.align_weights_i_to_j(
                    list(range(self._n_classes)),
                    list(range(self._n_classes, self._total_n_classes))
                )
        else:
            self._preprocessing = None

        self._network.eval()
        logger.info("Evaluating model...")
        for input_dict in loader:
            with torch.no_grad():
                if self._preprocessing is None or not has_trained:
                    logits = self._network(input_dict["inputs"].to(self._device))["logits"]
                else:
                    features = self._network.convnet(
                        input_dict["inputs"].to(self._device)
                    )[self.gmmn_config.get("features_key", "raw_features")]
                    if self.fakeclassifier_config:
                        features = self._preprocessing.transform(features)

                    if self._svm is None:
                        logits = self._network.classifier(features)["logits"]
                    else:
                        preds = self._svm.predict(features.cpu().numpy())
                        nb_classes = self._network.classifier.weights.shape[1]
                        logits = np.zeros((len(preds), nb_classes))
                        logits[np.arange(len(logits)), preds] = 1.0

            ytrue.append(input_dict["targets"])
            ypred.append(logits)
        self._network.train()

        if self._svm is not None:
            ypred = np.concatenate(ypred)
        else:
            ypred = torch.cat(ypred)
            ypred = F.softmax(ypred, dim=1)
            ypred = ypred.cpu().numpy()
        ytrue = torch.cat(ytrue).numpy()

        if self._task != self._n_tasks - 1 and self.fakeclassifier_config:
            if self.fakeclassifier_config.get("threshold") is not None:
                threshold1, threshold2 = self.fakeclassifier_config.get("threshold")
                logger.info(f"Using threshold ({threshold1}, {threshold2}).")
                maxes = ypred[..., :self._n_classes].max(axis=1)
                logger.info(f"Best confidence mean={maxes.mean()}, max={maxes.max()}.")
                ypred[maxes < threshold1, :self._n_classes] = 0.
                ypred[maxes > threshold2, self._n_classes:] = 0.
            elif self.fakeclassifier_config.get("bias") is not None:
                bias = self.fakeclassifier_config.get("bias")
                logger.info(f"Using bias {bias}.")
                ypred[..., :self._n_classes] += bias

        if self.fakeclassifier_config and self._task != self._n_tasks - 1:
            if self.fakeclassifier_config.get("keep_weights", "") == "all":
                logger.info("Keeping finetuned weights")
                self._network.classifier._weights = nn.ParameterList(
                    [self._network.classifier._weights[0]]
                )
            if self.fakeclassifier_config.get("keep_weights", "") == "all_not_first":
                if self._task == 0:
                    if self.fakeclassifier_config.get("what_post"):
                        self._network.post_processor = postprocessor
                    self._network.classifier._weights = real_clf_weights
                else:
                    logger.info("Keeping finetuned weights")
                    self._network.classifier._weights = nn.ParameterList(
                        [self._network.classifier._weights[0]]
                    )
            else:
                if self.fakeclassifier_config.get("what_post"):
                    self._network.post_processor = postprocessor
                self._network.classifier._weights = real_clf_weights
                if hasattr(
                    self._network.classifier, "_bias"
                ) and self._network.classifier._bias is not None:
                    self._network.classifier._bias = real_clf_bias

        self.train()
        return ypred, ytrue

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
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks,
            disable_progressbar=self._disable_progressbar
        )

    def train_supervised(self, train_loader, val_loader):
        logger.info("Finetuning ConvNet and classifier")

        if not isinstance(self.supervised_config, list):
            self.supervised_config = [self.supervised_config]

        for config in self.supervised_config:
            if not config["first_task"] and self._task == 0:
                continue
            if config.get("only_first_task") and self._task != 0:
                continue
            if config.get("min_task", 0) > self._task:
                continue

            if config.get("sampling", "none") == "undersample":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())
            else:
                loader = train_loader

            if config.get("class_weights"):
                logger.info("Computing class weights")
                self._class_weights = self.get_class_weights(loader)
            else:
                self._class_weights = None

            if config.get("update_gmmn", False) and self._task not in (0, self._n_tasks - 1):
                self.train_gmmn()
                self._setup_constraints()
            if config.get("update_constraints", False) and self._task not in (0, self._n_tasks - 1):
                self._setup_constraints()
            if config.get("update_sdc", False) and self._task not in (0, self._n_tasks - 1):
                logger.info("Update SDC")
                old_features, _ = utils.extract_features(self._old_model, loader)
                new_features, targets = utils.extract_features(self._network, loader)
                drift = losses.semantic_drift_compensation(
                    torch.tensor(old_features).to(self._device),
                    torch.tensor(new_features).to(self._device),
                    torch.tensor(targets).to(self._device)
                )
                with torch.no_grad():
                    self._ghosts = (self._ghosts[0] + drift, self._ghosts[1])
                    if self._network.classifier._negative_weights is not None:
                        self._network.classifier._negative_weights = self._network.classifier._negative_weights + drift
            if config.get("del_neg_weights", False):
                logger.info("Disabling neg weights & ghosts.")
                self._network.classifier.use_neg_weights = False

            optimizer, scheduler = self.setup_training(config)
            loops.single_loop(
                loader,
                val_loader,
                self._multiple_devices,
                self._network,
                config["epochs"],
                optimizer,
                scheduler=scheduler,
                train_function=self.forward_supervised,
                eval_function=self._accuracy,
                task=self._task,
                n_tasks=self._n_tasks,
                config=config,
                disable_progressbar=self._disable_progressbar
            )

            if config.get("align_weights") and self._task > 0:
                self._network.classifier.align_weights()
            if config.get("del_neg_weights", False):
                logger.info("Re-enabling neg weights & ghosts.")
                self._network.classifier.use_neg_weights = True

    def train_gmmn(self):
        logger.info("Training generator GMMN")
        config = self.gmmn_config

        # Fucking ugly, do something about it!
        if config.get("only_first") and self._task == 0:
            to_train = True
        elif config.get("only_first") and self._task > 0:
            to_train = False
        else:
            to_train = True

        if to_train:
            if config.get("reinit", "always") == "always" \
               or (config.get("reinit", "first") and self._task == 0):
                logger.info("Reinit GMMN")
                self._word_embeddings = Word2vec(
                    **self._args_we, data_path=self._w2v_path, device=self._device
                )
            elif config.get("reinit") not in ("always", "first"):
                raise NotImplementedError(f"Unknown value for GMMN: {config.get('reinit')}.")

        optimizer = factory.get_optimizer(
            [{
                "params": self._word_embeddings.parameters(),
                "lr": config["lr"]
            }], config["optimizer"], config["lr"], config.get("weight_decay", self._weight_decay)
        )

        if config.get("preprocessing"):
            if isinstance(config["preprocessing"], list):
                self._preprocessing = Scaler(config["preprocessing"])
            elif config["preprocessing"] == "robust":
                self._preprocessing = Scaler((0, 1), robust=1)
            elif config["preprocessing"] == "robust_scaled":
                self._preprocessing = Scaler((0, 1), robust=2)
            elif config["preprocessing"] == "normalize":
                self._preprocessing = Scaler((0, 1), normalize=True)
            elif config["preprocessing"] == "normalize_only":
                self._preprocessing = Scaler((0, 1), robust=-1, normalize=True)
            elif config["preprocessing"] == "normalize_truncate":
                self._preprocessing = Scaler((0, 1), robust=-1, normalize=True, truncate=True)
            elif config["preprocessing"] == "l2":
                self._preprocessing = Normalizer()
            else:
                raise ValueError(f"Unknown preprocessing: {config['preprocessing']}.")

        self._visual_features, self._visual_targets = loops.perclass_loop(
            self.inc_dataset,
            list(range(0, self._n_classes)),  # All seen classes
            self._multiple_devices,
            config["epochs"] if to_train else 0,
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
            batch_size=config.get("batch_size", 128),
            preprocessing=self._preprocessing,
            memory_class_ids=[]
            if self._task == 0 else list(range(self._n_classes - self._task_size)),
            memory=self.get_memory(),
            features_key=config.get("features_key", "raw_features")
        )

        if config.get("linear"):
            self._word_embeddings.eval()
            self._word_embeddings.add_linear_transform(bias=config["linear"]["bias"])
            self._word_embeddings.linear_transform.train()

            loops.linear_loop(
                self._visual_features,
                self._visual_targets,
                self._multiple_devices,
                config["linear"]["epochs"] if to_train else 0,
                factory.get_optimizer(
                    [
                        {
                            "params": self._word_embeddings.linear_transform.parameters(),
                            "lr": config["linear"]["lr"]
                        }
                    ], config["linear"]["optimizer"], config["linear"]["lr"],
                    config.get("weight_decay", self._weight_decay)
                ),
                self.forward_gmmn_linear,
                self._task,
                self._n_tasks,
                word_embeddings=self._word_embeddings,
                target_to_word=self.get_class_label,
                disable_progressbar=self._disable_progressbar,
                scheduler=factory.get_lr_scheduler(
                    config["linear"].get("scheduling"),
                    optimizer,
                    nb_epochs=config["linear"]["epochs"]
                ),
                batch_size=config["linear"].get("batch_size", 128),
                normalize=config["linear"].get("normalize", False)
            )

    def train_autoencoder(self):
        logger.info("Training generator Adverserial AutoEncoder")
        config = self.autoencoder_config

        # Fucking ugly, do something about it!
        if config.get("only_first") and self._task == 0:
            to_train = True
        elif config.get("only_first") and self._task > 0:
            to_train = False
        else:
            to_train = True

        if to_train:
            if config.get("reinit", "always") == "always" \
               or (config.get("reinit", "first") and self._task == 0):
                logger.info("Reinit AdvAutoEncoder")
                self._autoencoder = AdvAutoEncoder(**self._args_ae, device=self._device)
            elif config.get("reinit") not in ("always", "first"):
                raise NotImplementedError(f"Unknown value for AdvAE: {config.get('reinit')}.")

        if config.get("preprocessing"):
            if isinstance(config["preprocessing"], list):
                self._preprocessing = Scaler(config["preprocessing"])
            elif config["preprocessing"] == "l2":
                self._preprocessing = Normalizer()
            else:
                raise ValueError(f"Unknown preprocessing: {config['preprocessing']}.")

        self._visual_features, self._visual_targets = loops.adv_autoencoder_loop(
            self.inc_dataset,
            list(range(0, self._n_classes)),  # All seen classes
            self._multiple_devices,
            config["epochs"] if to_train else 0,
            self._task,
            self._n_tasks,
            network=self._network,
            autoencoder=self._autoencoder,
            target_to_word=self.get_class_label,
            disable_progressbar=self._disable_progressbar,
            batch_size=config.get("batch_size", 128),
            preprocessing=self._preprocessing,
            memory_class_ids=[]
            if self._task == 0 else list(range(self._n_classes - self._task_size)),
            memory=self.get_memory()
        )

    def train_fake_classifier(self, config):
        logger.info("Finetuning ConvNet and classifier")

        if config.get("adversarial_classifier"):
            self._domain_classifier = DomainClassifier(
                self._network.convnet.out_dim, device=self._device
            )
            optimizer, scheduler = self.setup_training(
                config,
                additional_parameters={"domain_classifier": self._domain_classifier.parameters()}
            )
        else:
            self._domain_classifier = None
            optimizer, scheduler = self.setup_training(config)

        logger.info("Finetuning fake classifier with offline generation.")

        if self._word_embeddings is not None:
            self._word_embeddings.eval()
        else:
            self._autoencoder.eval()

        if "look_ahead" in config:
            max_class = config.get("look_ahead") + self._n_classes
        else:
            max_class = self._total_n_classes

        if isinstance(config["nb_samples"], int):
            nb_samples = config["nb_samples"]
        else:
            nb_samples = int(torch.bincount(self._visual_targets).float().mean().cpu().item())
            logger.info(f"Gen {nb_samples} based on mean bincount.")

        fake_features, fake_targets = [], []
        for class_id in range(self._n_classes, max_class):
            targets = [class_id for _ in range(nb_samples)]
            cifar_targets = self.get_class_label(targets).to(self._device)
            with torch.no_grad():
                if self._word_embeddings is not None:
                    fake_features.append(self._word_embeddings(cifar_targets))
                else:
                    fake_features.append(self._autoencoder.generate(cifar_targets))
            fake_targets.append(torch.tensor(targets).long().to(self._device))

        fake_features = torch.cat(fake_features)

        if self._word_embeddings is not None:
            self._word_embeddings.train()
        else:
            self._autoencoder.train()
        if config.get("preprocessing"):
            logger.info("fake features preprocessing")
            if isinstance(config["preprocessing"], list):
                self._gen_preprocessing = Scaler(config["preprocessing"])
                fake_features = self._gen_preprocessing.fit_transform(fake_features)
            elif config["preprocessing"] == "l2":
                self._gen_preprocessing = Normalizer()
                fake_features = self._gen_preprocessing.fit_transform(fake_features)
            elif config["preprocessing"] == "reuse":
                self._gen_preprocessing = self._preprocessing
                fake_features = self._preprocessing.transform(fake_features)
            else:
                raise ValueError(f"Unknown preprocessing: {config['preprocessing']}.")

        features = torch.cat([self._visual_features, fake_features])
        targets = torch.cat([self._visual_targets.to(self._device), *fake_targets])
        flags = torch.cat((torch.ones(len(self._visual_features)), torch.zeros(len(fake_features))))

        if not isinstance(config.get("class_weights"), str) and config.get("class_weights") is True:
            logger.info("Computing class weights.")
            np_targets = targets.cpu().numpy()
            unique_targets = np.unique(np_targets)
            class_weights = compute_class_weight('balanced', unique_targets, np_targets)
            self._class_weights_fake = torch.tensor(class_weights).to(self._device).float()
        elif config.get("class_weights", "") == "mean":
            logger.info("Computing class weights MEAN mode.")
            np_targets = self._visual_targets.cpu().numpy()
            unique_targets = np.unique(np_targets)
            class_weights = compute_class_weight('balanced', unique_targets, np_targets)
            class_weights_unseen = np.ones(max_class - self._n_classes) * class_weights.mean()
            class_weights = np.concatenate((class_weights, class_weights_unseen))
            self._class_weights_fake = torch.tensor(class_weights).to(self._device).float()
        elif config.get("class_weights", "") == "one":
            logger.info("Computing class weights ONE mode.")
            np_targets = self._visual_targets.cpu().numpy()
            unique_targets = np.unique(np_targets)
            class_weights = compute_class_weight('balanced', unique_targets, np_targets)
            class_weights_unseen = np.ones(max_class - self._n_classes)
            class_weights = np.concatenate((class_weights, class_weights_unseen))
            self._class_weights_fake = torch.tensor(class_weights).to(self._device).float()
        else:
            self._class_weights_fake = None

        if config.get("svm"):
            logger.info("Learning SVM")
            from sklearn.svm import SVC
            self._svm = SVC(**config.get("svm"))
            self._svm.fit(features.cpu().numpy(), targets.cpu().numpy())
        else:
            if config.get("next_epochs") and self._task > 0:
                epochs = config["next_epochs"]
            else:
                epochs = config["epochs"]

            loops.features_to_classifier_loop(
                features,
                targets,
                flags,
                epochs,
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
        inputs = inputs.to(self._device)
        loss, outputs = losses.unsupervised_rotations(inputs, memory_flags, training_network)
        metrics["rot"] += loss.item()

        for i in range(len(outputs["attention"])):
            outputs["attention"][i] = outputs["attention"][i][:len(inputs)]

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                attention_loss = factor * losses.pod(
                    old_outputs["attention"],
                    outputs["attention"],
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._pod_spatial_config
                )
                loss += attention_loss
                metrics["att"] += attention_loss.item()
            if self._pod_flat_config:
                factor = self._pod_flat_config.get("factor", 1.)

                if self._pod_flat_config.get("scheduled", False):
                    factor = factor * math.sqrt(self._n_classes / self._task_size)

                distil_loss = factor * losses.embeddings_similarity(
                    outputs["raw_features"], old_outputs["raw_features"]
                )
                loss += distil_loss
                self._metrics["flat"] += distil_loss.item()

        return loss

    def forward_supervised(
        self, training_network, inputs, targets, memory_flags, metrics, epoch, epochs, config,
        **kwargs
    ):
        inputs = inputs.to(self._device)
        if config.get("features_process") is not None:
            # Create once the class for god sake
            scaler = Scaler(config.get("features_process"))
        else:
            scaler = None

        loss = 0.

        if self._task > 0 and self.ghost_config and self.ghost_config.get(
            "negative_weights_percent"
        ) and not config.get("del_neg_weights", False):
            percent = self.ghost_config["negative_weights_percent"]

            if self._task == self._n_tasks - 1:
                percent = self.ghost_config.get("negative_weights_percent_last", percent)

                if isinstance(percent, str) and percent == "hardest":
                    percent = 0.

            if isinstance(percent, str) and percent == "hardest":
                ghost_targets = self.get_class_label(
                    list(range(self._n_classes + self._task_size, self._total_n_classes))
                )
                ghost_targets = ghost_targets.to(self._device).long()

                ghost_similarities = self.similarity_matrix[targets.to(
                    self._device
                )].index_select(dim=1, index=ghost_targets)
                if len(ghost_targets) == 0:
                    additional_features = None
                    ghost_flags = None
                    ghost_batch_size = None
                else:
                    most_similars = ghost_targets[ghost_similarities.max(dim=1)[1]]
                    most_similars = self.get_inv_class_label(most_similars).to(self._device)

                    additional_features = []
                    for real_t, ghost_t in zip(targets, most_similars):
                        indexes = self._ghosts[1] == ghost_t
                        sub_features = self._ghosts[0][indexes]
                        rnd_index = torch.randint(low=0, high=len(sub_features), size=(1,))[0]

                        additional_features.append(sub_features[rnd_index])

                    additional_features = torch.stack(additional_features)
                    targets = torch.cat((targets.cpu(), most_similars.cpu()))
                    ghost_flags = torch.cat(
                        (torch.ones(len(additional_features)), torch.zeros(len(targets)))
                    ).to(self._device)
                    ghost_batch_size = len(additional_features)
            elif percent == 0.:
                additional_features = None
                ghost_flags = None
                ghost_batch_size = None
            else:
                batch_size = len(inputs)
                ghost_batch_size = max(int(batch_size * percent), 1)
                indexes = torch.randperm(len(self._ghosts[0]))[:ghost_batch_size]
                additional_features = self._ghosts[0][indexes]
                targets = torch.cat((targets, self._ghosts[1][indexes].cpu()), 0)

                ghost_flags = torch.cat(
                    (torch.ones(batch_size), torch.zeros(len(additional_features)))
                ).to(self._device)
        else:
            additional_features = None
            ghost_flags = None
            ghost_batch_size = None

        outputs = training_network(
            inputs, features_processing=scaler, additional_features=additional_features
        )

        if self._task >= 2 and self.placement_config and config.get("ghost_placement"):
            # Starting from third tasks, we should see the effect of the ghost
            # on the classes they were mimicking. Here we want to enforce the new
            # classes to place themselves in the empty space given by the ghost.
            features = outputs[self.gmmn_config.get("features_key", "raw_features")]

            placement_loss = losses.similarity_per_class(
                features,
                targets if ghost_batch_size is None else targets[:-ghost_batch_size],
                *self._old_ghosts,
                epoch=epoch,
                epochs=epochs,
                memory_flags=memory_flags,
                old_centroids_features=self._c_features,
                old_centroids_targets=self._c_targets,
                **self.placement_config
            )
            if not isinstance(placement_loss, float):
                metrics["plc"] += placement_loss.item()
            loss += placement_loss

            if config.get("only_ghost_placement", False):
                return loss
        elif self._task >= 2 and self.adv_placement_config and config.get("ghost_placement"):
            real_features = outputs[self.gmmn_config.get("features_key", "raw_features")]
            real_features = real_features[~memory_flags.bool()]

            if len(real_features) == 0:
                # Batch is made only of memory data, rare but can happen.
                loss += 0.
            else:
                ghost_features = self._old_ghosts[0]
                real_targets = torch.ones(len(real_features)).float().to(self._device)
                ghost_targets = torch.zeros(len(ghost_features)).float().to(self._device)

                domain_features = torch.cat((real_features, ghost_features))
                domain_targets = torch.cat((real_targets, ghost_targets))

                domain_logits = self._network.domain_classifier(domain_features)

                adv_plc_loss = F.binary_cross_entropy_with_logits(
                    domain_logits.squeeze(1), domain_targets
                )
                adv_plc_loss = self.adv_placement_config["factor"] * adv_plc_loss
                if self.adv_placement_config["scheduled"]:
                    adv_plc_loss = (1 - epoch / epochs) * adv_plc_loss

                metrics["advPlc"] += adv_plc_loss.item()
                loss += adv_plc_loss

                if config.get("only_ghost_placement", False):
                    return loss

        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self.ghost_config:
                nca_config.update(self.ghost_config.get("ams_loss", {}))
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss += losses.nca(
                outputs["logits"],
                targets,
                class_weights=self._class_weights,
                memory_flags=ghost_flags,
                **nca_config
            )
            metrics["ams"] += loss.item()
        elif self._softmax_ce:
            loss += F.cross_entropy(
                self._network.post_process(outputs["logits"]), targets.to(outputs["logits"].device)
            )
            metrics["cce"] += loss.item()
        else:
            raise ValueError("No classification loss defined!")

        if self._task > 0 and self.ucir_ranking_config:
            if ghost_batch_size is not None:
                r_logits = outputs["logits"][:-ghost_batch_size]
                r_targets = targets[:-ghost_batch_size]
            else:
                r_logits = outputs["logits"]
                r_targets = targets

            ranking_loss = self.ucir_ranking_config.get("factor", 1.0) * losses.ucir_ranking(
                r_logits, r_targets.to(r_logits.device), self._n_classes, self._task_size
            )
            metrics["rnk"] += ranking_loss.item()
            loss += ranking_loss

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                attention_loss = factor * losses.pod(
                    old_outputs["attention"],
                    outputs["attention"],
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._pod_spatial_config
                )
                loss += attention_loss
                metrics["att"] += attention_loss.item()

            if self._pod_flat_config:
                factor = self._pod_flat_config.get("factor", 1.)

                if self._pod_flat_config.get("scheduled", False):
                    factor = factor * math.sqrt(self._n_classes / self._task_size)

                distil_loss = factor * losses.embeddings_similarity(
                    outputs["raw_features"], old_outputs["raw_features"]
                )
                loss += distil_loss
                metrics["flat"] += distil_loss.item()

        if self._task != 0 and self._task != self._n_tasks - 1:
            if self.hyperplan_config:
                type_ = self.hyperplan_config.get("type", "ortho_abs")
                factor = self.hyperplan_config["factor"]
                if self.hyperplan_config.get("scheduled_factor", False):
                    factor = factor * math.sqrt(self._total_n_classes - self._n_classes)

                features = outputs[self.gmmn_config.get("features_key", "raw_features")]

                if not self.hyperplan_config.get("apply_on_new", True):
                    # Only applying on memory samples
                    old_classes = list(range(self._n_classes - self._task_size))
                    indexes = np.where(np.isin(targets.cpu(), old_classes))[0]
                    features = features[indexes]

                if len(features) == 0:
                    # Can happen if we don't apply the reg on new classes and
                    # that the batch has not a single memory sample.
                    metrics["hyper"] += 0.
                else:
                    if self.hyperplan_config.get("normalize_features", True):
                        features = F.normalize(features, dim=1, p=2)

                    if self._svm_weights is None:
                        simi = torch.mm(features, self._hyperplan.T)
                        if self.hyperplan_config.get("add_bias", False):
                            simi = simi + self._hyperplan_bias
                        simi = simi.view(-1)
                    else:
                        simi = []
                        for sv, gamma, dual_coef, intercept in zip(
                            self._svm_weights["sv"], self._svm_weights["gamma"],
                            self._svm_weights["dual_coef"], self._svm_weights["intercept"]
                        ):
                            diff = sv[None, :, :] - features[:, None, :]
                            tmp = torch.exp(-gamma * diff.norm(dim=-1)**2)
                            dec = dual_coef.mm(tmp.T) - intercept
                            simi.append(dec.view(-1))
                        simi = torch.cat(simi)

                    # simi should be in [-1 + b, +1 + b]

                    if type_ == "anticorrelation":
                        hinges = torch.clamp(simi + 1., min=0.)
                    elif type_ == "anticorrelation_neg":
                        hinges = torch.clamp(simi, min=0.)
                    elif type_ in ("need_neg", "boundary"):
                        # Check if the point is one the correct side of the hyperplan
                        hinges = torch.clamp(simi, min=0)
                    elif type_ == "support_vectors":
                        # Check if the point is beyond its class support vectors
                        hinges = torch.clamp(simi + 1, min=0)
                    else:
                        raise NotImplementedError(f"Unknow type {type_}.")

                    if self.hyperplan_config.get("adamine"):
                        nb_not_null = torch.nonzero(torch.clamp(hinges - 1e-6, min=0.)).shape[0]
                        nb_not_null = max(nb_not_null, 1)
                        hyper_loss = torch.sum(hinges) / nb_not_null
                    else:
                        hyper_loss = torch.mean(hinges)

                    hyper_loss = factor * hyper_loss

                    if self.hyperplan_config.get("scheduled"):
                        hyper_loss = hyper_loss * math.sqrt(self._n_classes / self._task_size)

                    metrics["hyper"] += hyper_loss.item()
                    loss += hyper_loss
            elif self.ghost_config and self.ghost_config.get("factor"):
                features = outputs[self.gmmn_config.get("features_key", "raw_features")]
                ghost_reg = ghost_semantic_regularization(
                    features, targets, *self._ghosts, self.similarity_matrix, **self.ghost_config
                )
                if self.ghost_config.get("scheduled_factor", False):
                    factor = math.sqrt(self._total_n_classes - self._n_classes)
                    ghost_reg = factor * ghost_reg

                metrics["gho"] += ghost_reg.item()
                loss += ghost_reg

        if self.real_config:
            features = outputs[self.gmmn_config.get("features_key", "raw_features")]
            real_reg = semantic_regularization(
                features, targets, self.similarity_matrix, **self.real_config
            )
            metrics["rea"] += real_reg.item()
            loss += real_reg

        if torch.isnan(loss):
            raise ValueError(f"Nan loss in {str(metrics)}")

        return loss

    def forward_gmmn(self, visual_features, semantic_features, class_id, words, metrics):
        loss = mmd(real=visual_features, fake=semantic_features, **self.gmmn_config["mmd"])

        if self.gmmn_config.get("old_mmd") and self._old_word_embeddings is not None:
            old_unseen_limit = self._n_classes - self._task_size

            if not self.gmmn_config["old_mmd"].get(
                "apply_unseen", False
            ) and class_id >= old_unseen_limit:
                return loss
            with torch.no_grad():
                old_semantic_features = self._old_word_embeddings(words)

            factor = self.gmmn_config["old_mmd"]["factor"]
            _type = self.gmmn_config["old_mmd"].get("type", "mmd")
            if _type == "mmd":
                old_loss = factor * mmd(
                    real=old_semantic_features, fake=semantic_features, **self.gmmn_config["mmd"]
                )
            elif _type == "kl":
                old_loss = factor * F.kl_div(
                    semantic_features, old_semantic_features, reduction="batchmean"
                )
            elif _type == "l2":
                old_loss = factor * torch.pairwise_distance(
                    semantic_features, old_semantic_features, p=2
                ).mean()
            elif _type == "cosine":
                old_loss = factor * (
                    1 - torch.cosine_similarity(semantic_features, old_semantic_features)
                ).mean()
            else:
                raise ValueError(f"Unknown distillation: {_type}.")

            if self.gmmn_config.get("scheduled"):
                old_loss = old_loss * math.sqrt(self._n_classes / self._task_size)

            metrics["old"] += old_loss.item()
            return loss + old_loss
        return loss

    def forward_gmmn_linear(self, visual_features, semantic_features):
        return F.mse_loss(visual_features, semantic_features)

    def forward_fakeclassifier(self, logits, targets, flags, metrics):
        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            nca_config.update(self.fakeclassifier_config.get("loss", {}))
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            try:
                loss = losses.nca(
                    logits,
                    targets,
                    class_weights=self._class_weights_fake,
                    memory_flags=flags,
                    **nca_config,
                )
            except:
                breakpoint()
        elif self._softmax_ce:
            loss = F.cross_entropy(self._network.post_process(logits), targets)
        else:
            raise ValueError("No classification loss defined!")
        metrics["clf"] += loss.item()

        if self._domain_classifier is not None:
            weights = self._network.classifier.weights
            domain_logits = self._domain_classifier(weights)

            nb_unseen = self._total_n_classes - self._n_classes
            domain_targets = torch.ones(self._total_n_classes).float()
            domain_targets[-nb_unseen:] = 0
            factor = self.fakeclassifier_config["training"]["adversarial_classifier"]["factor"]
            domain_loss = factor * F.binary_cross_entropy_with_logits(
                domain_logits.view(-1), domain_targets.to(domain_logits.device)
            )

            metrics["adv"] += domain_loss.item()
            loss += domain_loss

        return loss

    def get_class_weights(self, loader):
        targets = []
        for input_dict in loader:
            targets.append(input_dict["targets"])
        targets = torch.cat(targets).cpu().numpy()
        unique_targets = np.unique(targets)
        class_weights = compute_class_weight('balanced', unique_targets, targets)
        return torch.tensor(class_weights).to(self._device).float()

    def get_class_weights_raw(self, targets):
        unique_targets = np.unique(targets)
        class_weights = compute_class_weight('balanced', unique_targets, targets)
        return torch.tensor(class_weights).to(self._device).float()

    # -----------
    # Constraints
    # -----------

    def _setup_constraints(self):
        if self._word_embeddings is None:
            return

        self.similarity_matrix = generate_similarity_matrix(
            self._word_embeddings, self.get_class_label(list(range(self._total_n_classes)))
        )

        if self.ghost_config:
            self._word_embeddings.eval()
            if self._ghosts is not None:
                self._old_ghosts = copy.deepcopy(self._ghosts)
            self._ghosts = self._gen_ghost_features(self.ghost_config)
            self._word_embeddings.train()

        if self.hyperplan_config:
            assert self.ghost_config
            if self.hyperplan_config.get("linear_nn"):
                self._hyperplan = self._gen_linear_hyperplan_constraints(
                    self._ghosts[0],
                    self._ghosts[1],
                    one_per_ghost=self.hyperplan_config.get("one_per_ghost", False),
                    epochs=self.hyperplan_config.get("epochs", 10),
                    class_weights=self.hyperplan_config.get("class_weights", False),
                    apply_current=self.hyperplan_config.get("apply_current", True),
                    flip=self.hyperplan_config.get("flip", False)
                )
                self._hyperplan_bias = None
            else:
                self._hyperplan, self._hyperplan_bias = self._gen_hyperplan_constraints(
                    self._ghosts[0],
                    self._ghosts[1],
                    C=self.hyperplan_config.get("C", 1.0),
                    end_normalize=self.hyperplan_config.get("end_normalize", False),
                    end_normalize_bias=self.hyperplan_config.get(
                        "end_normalize_bias", self.hyperplan_config.get("end_normalize", False)
                    ),
                    normalize_features=self.hyperplan_config.get("normalize_features", True),
                    one_per_ghost=self.hyperplan_config.get("one_per_ghost", False),
                    apply_current=self.hyperplan_config.get("apply_current", True),
                    flip=self.hyperplan_config.get("flip", False),
                    kernel=self.hyperplan_config.get("kernel", "linear")
                )
        else:
            self._hyperplan = None
            self._hyperplan_bias = None

    def _gen_linear_hyperplan_constraints(
        self,
        ghost_features,
        ghost_targets,
        one_per_ghost=False,
        epochs=10,
        class_weights=False,
        apply_current=True,
        flip=False
    ):
        logger.info("Generating linear hyperplan constraint.")
        if apply_current:  # Just previous tasks
            classes = list(range(self._n_classes - self._task_size, self._n_classes))
        else:
            classes = []

        _, loader = self.inc_dataset.get_custom_loader(classes, memory=self.get_memory())
        real_features = utils.extract_features(self._network, loader)[0]
        if flip:
            _, loader = self.inc_dataset.get_custom_loader(
                classes, memory=self.get_memory(), mode="flip"
            )
            real_features_flipped = utils.extract_features(self._network, loader)[0]
            real_features = np.concatenate((real_features, real_features_flipped))

        real_features = torch.tensor(real_features).float().to(self._device)
        real_targets = torch.zeros(len(real_features)).long().to(self._device)

        if one_per_ghost:
            hyperplans = []
            for target in torch.unique(ghost_targets):
                indexes = ghost_targets == target

                sub_f = ghost_features[indexes]
                sub_t = torch.ones((len(indexes))).long().to(self._device)

                clf = BinaryCosineClassifier(real_features.shape[1]).to(self._device)
                opt = torch.optim.Adam(clf.parameters(), lr=0.001)
                ghost_targets.fill_(1).long().to(self._device)

                features = torch.cat((real_features, sub_f))
                targets = torch.cat((real_targets, sub_t)).float().to(self._device)
                if class_weights:
                    cw = self.get_class_weights_raw(targets.cpu().numpy())
                else:
                    cw = None

                def func_loss(feats, t, fl, m):
                    if cw is None:
                        weight = None
                    else:
                        weight = cw[t.long()]
                    return F.binary_cross_entropy_with_logits(feats.squeeze(1), t, weight=weight)

                loops.features_to_classifier_loop(
                    features, targets, None, epochs, opt, clf, func_loss, disable_progressbar=True
                )

                w = F.normalize(clf.weight.data, dim=1, p=2)
                hyperplans.append(w)
            return torch.cat(hyperplans)
        else:
            clf = BinaryCosineClassifier(real_features.shape[1]).to(self._device)
            opt = torch.optim.Adam(clf.parameters(), lr=0.001)
            ghost_targets.fill_(1).long().to(self._device)

            features = torch.cat((real_features, ghost_features))
            targets = torch.cat((real_targets, ghost_targets)).float().to(self._device)
            if class_weights:
                cw = self.get_class_weights_raw(targets.cpu().numpy())
            else:
                cw = None

            def func_loss(feats, t, fl, m):
                if cw is None:
                    weight = None
                else:
                    weight = cw[t.long()]
                return F.binary_cross_entropy_with_logits(feats.squeeze(1), t, weight=weight)

            loops.features_to_classifier_loop(
                features, targets, None, epochs, opt, clf, func_loss, disable_progressbar=True
            )

            return F.normalize(clf.weight.data, dim=1, p=2)

    def _gen_hyperplan_constraints(
        self,
        ghost_features,
        ghost_targets,
        C=1.0,
        end_normalize=False,
        end_normalize_bias=False,
        normalize_features=True,
        one_per_ghost=False,
        apply_current=True,
        flip=False,
        kernel="linear"
    ):
        self._svm_weights = None

        logger.info("Generating hyperplan constraint.")
        if apply_current:  # Just previous task
            classes = list(range(self._n_classes - self._task_size, self._n_classes))
        else:
            classes = []

        _, loader = self.inc_dataset.get_custom_loader(classes, memory=self.get_memory())
        real_features = utils.extract_features(self._network, loader)[0]
        if flip:
            _, loader = self.inc_dataset.get_custom_loader(
                classes, memory=self.get_memory(), mode="flip"
            )
            real_features_flipped = utils.extract_features(self._network, loader)[0]
            real_features = np.concatenate((real_features, real_features_flipped))
        real_targets = np.zeros(len(real_features))

        ghost_features = ghost_features.cpu().numpy()

        if one_per_ghost:
            ghost_targets = ghost_targets.cpu().numpy()
            if kernel == "linear":
                hyperplans, biases = [], []
            else:
                self._svm_weights = collections.defaultdict(list)

            for class_id in np.unique(ghost_targets):
                tmp_ghost_features = ghost_features[np.where(ghost_targets == class_id)[0]]
                tmp_features = np.concatenate((real_features, tmp_ghost_features))
                if normalize_features:
                    tmp_features = tmp_features / np.linalg.norm(
                        tmp_features, axis=1, keepdims=True
                    )

                tmp_targets = np.concatenate((real_targets, np.ones(len(tmp_ghost_features))))

                svm = SVC(C=C, kernel=kernel, gamma="scale")
                svm.fit(tmp_features, tmp_targets)

                if kernel == "linear":
                    hyperplans.append(torch.tensor(svm.coef_[0]).float().to(self._device)[None])
                    biases.append(torch.tensor(svm.intercept_[0]).float().to(self._device))
                else:
                    self._svm_weights["sv"].append(
                        torch.tensor(svm.support_vectors_).float().to(self._device)
                    )
                    self._svm_weights["gamma"].append(
                        1 / (tmp_features.shape[1] * tmp_features.var())
                    )
                    self._svm_weights["dual_coef"].append(
                        torch.tensor(svm.dual_coef_).float().to(self._device)
                    )
                    self._svm_weights["intercept"].append(
                        torch.tensor(svm.intercept_).float().to(self._device)
                    )

            if kernel == "linear":
                hyperplan = torch.cat(hyperplans)
                bias = torch.stack(biases)
            else:
                hyperplan, bias = None, None
        else:
            ghost_targets = np.ones(len(ghost_features))

            features = np.concatenate((real_features, ghost_features))
            if normalize_features:
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
            targets = np.concatenate((real_targets, ghost_targets))

            svm = SVC(C=C, kernel=kernel, gamma="scale")
            svm.fit(features, targets)
            acc = svm.score(features, targets)
            logger.info(f"SVM got {acc} on the train set (binary, real vs ghost).")

            if kernel == "linear":
                hyperplan = torch.tensor(svm.coef_[0]).float().to(self._device)[None]
                bias = torch.tensor(svm.intercept_[0]).float().to(self._device)
            else:
                self._svm_weights = {
                    "sv":
                        [torch.tensor(svm.support_vectors_).float().to(self._device)],
                    "gamma":
                        [torch.tensor([1 / (features.shape[1] * features.var())]).float().to(self._device)],
                    "dual_coef":
                        [torch.tensor(svm.dual_coef_).float().to(self._device)],
                    "intercept":
                        [torch.tensor(svm.intercept_).float().to(self._device)]
                }
                hyperplan, bias = None, None

        if end_normalize:
            hyperplan = F.normalize(hyperplan, dim=1, p=2)
        if end_normalize_bias:
            if len(bias.shape) > 1:
                bias = F.normalize(bias, dim=1, p=2)
            else:
                bias = F.normalize(bias, dim=0, p=2)

        return hyperplan, bias

    def _gen_ghost_features(self, config):
        if config.get("nb_unseen_classes") is not None:
            classes_to_create = list(
                range(self._n_classes, self._n_classes + config["nb_unseen_classes"])
            )
        else:
            classes_to_create = list(range(self._n_classes, self._total_n_classes))

        if config.get("cheat"):
            logger.info("Custom cheat for ghosts")
            # Test if real future taken in the past are really better than our fakes
            # This should be our upperbound

            if config["cheat"] == "pixels":
                logger.info("Cheat at pixel-level")
                _, loader = self.inc_dataset.get_custom_loader(classes_to_create)
                x, y, m = loader.dataset.x, loader.dataset.y, loader.dataset.memory_flags
                self._cheat_pixels = (x, y, m)

                f, t = utils.extract_features(self._network, loader)
                features = torch.tensor(f).to(self._device)
                targets = torch.tensor(t).long().to(self._device)
            else:
                if config["cheat"] == "own":
                    martymcfly = self._network
                else:
                    martymcfly = network.BasicNet(
                        self.args["convnet"],
                        convnet_kwargs=self.args.get("convnet_config", {}),
                        classifier_kwargs=self.args.get("classifier_config"),
                        postprocessor_kwargs=self.args.get("postprocessor_config", {}),
                        device=self._device,
                        extract_no_act=True,
                        classifier_no_act=self.args.get("classifier_no_act", True),
                        return_features=True,
                        attention_hook=True,
                        rotations_predictor=True
                    )
                    state_dict = torch.load(
                        os.path.join(config["cheat"], f"net_0_task_{self._task + 1}.pth")
                    )
                    for key in list(state_dict.keys()):
                        if "classifier" in key:
                            del state_dict[key]
                    martymcfly.load_state_dict(state_dict, strict=True)

                features, targets = [], []
                for class_id in classes_to_create:
                    _, loader = self.inc_dataset.get_custom_loader([class_id])
                    f, t = utils.extract_features(martymcfly, loader)

                    if config["amount_per_class"] is not None:
                        indexes = np.arange(len(f))
                        indexes = np.random.choice(indexes, size=config["amount_per_class"])
                        f = f[indexes]
                        t = t[indexes]
                    features.append(f)
                    targets.append(t)

                features = np.concatenate(features)
                targets = np.concatenate(targets)
                features = torch.tensor(features).to(self._device)
                targets = torch.tensor(targets).long().to(self._device)
        else:
            features, targets = [], []
            for class_id in classes_to_create:
                class_ids = [class_id for _ in range(config["amount_per_class"])]
                real_class_ids = self.get_class_label(class_ids).to(self._device)

                with torch.no_grad():
                    features.append(self._word_embeddings(real_class_ids))

                targets.extend(class_ids)

            features = torch.cat(features).to(self._device)
            targets = torch.tensor(targets).long().to(self._device)

            if config.get("inverse_transform") and self._preprocessing is not None:
                logger.info("Inverse transform of ghost features.")
                features = self._preprocessing.inverse_transform(features)

            if config.get("align_features_per_class"):
                logger.info("Aligning features per class")
                new_features, new_targets = [], []
                for t in torch.unique(targets):
                    indexes = t == targets
                    new_features.append(self._network.classifier.align_features(features[indexes]))
                    new_targets.append(targets[indexes])
                features = torch.cat(new_features)
                targets = torch.cat(new_targets)
            elif config.get("align_features"):
                logger.info("Aligning features")
                features = self._network.classifier.align_features(features)

        if config.get("average_per_class"):
            f, t = [], []
            for i in classes_to_create:
                indexes = targets == i
                f.append(features[indexes].mean(dim=0))
                t.append(i)
            avg_features = torch.stack(f)

        if config.get("subsample_per_class"):
            f, t = [], []
            for i in classes_to_create:
                indexes = np.where(targets.cpu().numpy() == i)[0]
                indexes = np.random.choice(
                    indexes, size=config["subsample_per_class"], replace=False
                )
                f.append(features[indexes])
                t.append(targets[indexes])
            features = torch.cat(f)
            targets = torch.cat(t)

        if config.get("negative_weights", False):
            self._network.classifier.set_negative_weights(
                avg_features, config["negative_weights_ponderation"]
            )

        return features, targets

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


def semantic_regularization(
    features, targets, similarity_matrix, margin=None, aggreg="mean", factor=1.0, metric="cosine"
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
    if metric == "cosine":
        similarities = F.cosine_similarity(left, right)

        if margin is not None:
            margins = torch.ones_like(similarities) * margin
        else:
            margins = similarity_matrix[targets[pair_indexes[..., 0]], targets[pair_indexes[...,
                                                                                            1]]]

        hinges = torch.clamp(similarities - margins, min=0.)

        return factor * _aggreg(hinges, aggreg, features_dim=features.shape[1])
    elif metric == "gor":
        similarities = torch.sum(torch.mul(left, right), 1)
        return factor * _aggreg(similarities, aggreg, features_dim=features.shape[1])
    elif metric == "snr":
        noise = left - right
        var_noise = noise.var(axis=1, unbiased=True)
        var_anchor = right.var(axis=1, unbiased=True)

        dist = torch.mean(var_anchor / var_noise)
        return factor * dist
    else:
        raise NotImplementedError(f"Unknown metric: {metric}.")


def ghost_semantic_regularization(
    features,
    targets,
    ghost_features,
    ghost_targets,
    similarity_matrix,
    margin=None,
    aggreg="mean",
    factor=1.0,
    metric="cosine",
    scale_real=False,
    scale_ghost=False,
    normalize=False,
    triplet=False,
    against_all=None,
    **kwargs
):
    if scale_real:
        scale(features, (0, 1))
    if scale_ghost:
        scale(ghost_features, (0, 1))

    if normalize:
        features = F.normalize(features, p=2, dim=-1)
        ghost_features = F.normalize(ghost_features, p=2, dim=-1)

    if triplet:
        # Anchor-positive distances
        dists = -torch.mm(features, features.T)
        indexes_not_equal = ~torch.eye(len(targets)).bool().to(features.device)
        labels_equal = targets.unsqueeze(0) == targets.unsqueeze(1)
        mask = indexes_not_equal & labels_equal.to(features.device)
        ap = (dists.to(features.device) * mask.to(features.device).float()).max(dim=1)[0]

        # Anchor-negative distances
        an = -torch.mm(features, ghost_features.T)
        an = an.min(dim=1)[0]

        hinges = torch.clamp(margin + ap - an, min=0.)
        return _aggreg(hinges, aggreg, features_dim=features.shape[1])
    elif against_all is not None:
        assert normalize
        if margin is None:
            margin = 0.

        similarities = torch.mm(features, ghost_features.T)
        if isinstance(against_all, int):
            similarities = similarities.topk(against_all, dim=1)[0]

        hinges = torch.clamp(similarities.view(-1) - margin, min=0.)
        return _aggreg(hinges, aggreg, features_dim=features.shape[1])
    else:
        neg_indexes = []
        np_targets = targets.cpu().numpy()
        for index, target in enumerate(np_targets):
            neg_index = np.random.choice(len(ghost_targets))
            neg_indexes.append(neg_index)

        selected_ghosts_features = ghost_features[neg_indexes]
        selected_ghosts_targets = ghost_targets[neg_indexes]

        if metric == "cosine":
            similarities = F.cosine_similarity(features, selected_ghosts_features)

            if margin is not None:
                margins = torch.ones_like(similarities) * margin
            else:
                margins = similarity_matrix[targets, selected_ghosts_targets]

            hinges = torch.clamp(similarities - margins, min=0.)

            return factor * _aggreg(hinges, aggreg, features_dim=features.shape[1])
        elif metric == "gor":
            similarities = torch.sum(torch.mul(features, selected_ghosts_features), 1)
            return factor * _aggreg(similarities, aggreg, features_dim=features.shape[1])
        elif metric == "snr":
            noise = selected_ghosts_features - features
            var_noise = noise.var(axis=1, unbiased=True)
            var_anchor = features.var(axis=1, unbiased=True)

            dist = torch.mean(var_anchor / var_noise)
            return factor * dist
        else:
            raise NotImplementedError(f"Unknown metric: {metric}.")


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
    classes = class_ids.clone().detach().long().to(word_embeddings.device)

    with torch.no_grad():
        embeddings = word_embeddings.forward(classes, only_word=True)

    embeddings = F.normalize(embeddings, dim=1, p=2)
    return torch.mm(embeddings, embeddings.t())


def mmd(
    fake,
    real,
    sigmas=[2, 5, 10, 20, 40, 80],
    normalize=True,
    scale_matrix=False,
    bucher=False,
    **kwargs
):
    """Maximum Mean Discrepancy with several Gaussian kernels."""
    if normalize:
        real = F.normalize(real, dim=1, p=2)
        #fake = F.normalize(fake, dim=1, p=2)

    if bucher:
        return moment_loss(fake, real, sigma=sigmas, device=real.device)

    xy = torch.cat((fake, real), dim=0)

    if scale_matrix:
        scale = get_scale_matrix(len(fake), len(real), real.device)
        scale = torch.matmul(scale, scale.t())
    else:
        scale = 1.

    xx = torch.mm(xy, xy.t())
    x2 = torch.sum(xx**2, dim=1, keepdim=True)

    exponent = xx - 0.5 * x2 - 0.5 * x2.t()

    loss = 0.
    for sigma in sigmas:
        kernel_val = torch.exp(exponent / sigma)
        loss += torch.sum(scale * kernel_val)
        if torch.isnan(loss):
            breakpoint()

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

    return torch.sqrt(loss)


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_sigmas(sigmas, device):
    sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
    return -1 / (2 * sigmas)


#@functools.lru_cache(maxsize=1, typed=False)
def get_scale_matrix(M, N, device):
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    return torch.cat((s1, s2), 0).to(device)


def scale(tensor, feature_range=(0, 1)):
    data_min = torch.min(tensor, dim=0)[0]
    data_max = torch.max(tensor, dim=0)[0]
    data_range = data_max - data_min

    # Handle null values
    data_range[data_range == 0.] = 1.

    scale_ = (feature_range[1] - feature_range[0]) / data_range
    min_ = feature_range[0] - data_min * scale_

    return tensor.mul(scale_).add_(min_)


class Scaler:
    """
    Transforms each channel to the range [a, b].
    """

    def __init__(self, feature_range, robust=0, normalize=False, truncate=False):
        self.feature_range = feature_range
        self.robust = robust
        self.normalize = normalize
        self.truncate = truncate

        if self.robust:
            self.skprepro = skpreprocessing.RobustScaler()

    def fit(self, tensor):
        if self.normalize:
            self.mu, self.sigma = tensor.mean(dim=0), tensor.std(dim=0)
            tensor = (tensor - self.mu.expand_as(tensor)) / self.sigma.expand_as(tensor)

        if self.truncate:
            tensor = tensor.clamp(min=self.feature_range[0], max=self.feature_range[1])

        if self.robust > 0:
            device = tensor.device
            tensor = tensor.cpu().numpy()
            tensor = self.skprepro.fit_transform(tensor)
            tensor = torch.tensor(tensor).to(device)

        if self.robust == 0 or self.robust == 2:
            data_min = torch.min(tensor, dim=0)[0]
            data_max = torch.max(tensor, dim=0)[0]
            data_range = data_max - data_min

            # Handle null values
            data_range[data_range == 0.] = 1.

            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
            self.min_ = self.feature_range[0] - data_min * self.scale_
            self.data_min_ = data_min
            self.data_max_ = data_max
            self.data_range_ = data_range

        return self

    def transform(self, tensor):
        if self.normalize:
            tensor = (tensor - self.mu.expand_as(tensor)) / self.sigma.expand_as(tensor)

        if self.robust > 0:
            device = tensor.device
            tensor = tensor.cpu().numpy()
            tensor = self.skprepro.transform(tensor)
            tensor = torch.tensor(tensor).to(device)

        if self.robust == 0 or self.robust == 2:
            return tensor.mul_(self.scale_).add_(self.min_)
        return tensor

    def inverse_transform(self, tensor):
        if self.normalize:
            tensor = (tensor * self.sigma.expand_as(tensor)) + self.mu.expand_as(tensor)

        if self.robust == 0 or self.robust == 2:
            tensor = tensor.sub_(self.min_).div_(self.scale_)
        if self.robust > 0:
            device = tensor.device
            tensor = tensor.cpu().numpy()
            tensor = self.skprepro.inverse_transform(tensor)
            tensor = torch.tensor(tensor).to(device)
        return tensor

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)


class Normalizer:

    def fit(self, tensor):
        return self

    def transform(self, tensor):
        return F.normalize(tensor, dim=1, p=2)

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)
