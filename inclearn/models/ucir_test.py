import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, losses, network, utils
from inclearn.models.icarl import ICarl


class UCIRTest(ICarl):
    """Implements Learning a Unified Classifier Incrementally via Rebalancing

    * http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
    """

    def __init__(self, args):
        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        if "distillation_loss" in args:
            import warnings
            warnings.warn("distillation_loss is replaced by less_forget")
            args["less_forget"] = args["distillation_loss"]
            del args["distillation_loss"]

        self._use_mimic_score = args.get("mimic_score", False)
        self._use_less_forget = args.get("less_forget", True)
        self._lambda_schedule = args.get("lambda_schedule", False)
        self._use_ranking = args.get("ranking_loss", False)
        self._scaling_factor = args.get("scaling_factor", True)

        self._use_relative_teachers = args.get("relative_teachers", False)
        self._relative_teachers_old = args.get("relative_teacher_on_memory", False)

        self._use_gor_reg = args.get("gor_reg", False)

        self._use_ams_ce = args.get("adaptative_margin_softmax", False)

        self._use_attention_residual = args.get("attention_residual", False)

        self._use_teacher_confidence = args.get("teacher_confidence", False)

        self._use_proxy_nca = args.get("proxy_nca", False)
        self._proxy_per_class = args.get("proxy_per_class", 1)

        self._use_npair = args.get("use_npair", False)
        self._use_mer = args.get("use_mer", False)

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._weights_orthogonality = args.get("weights_orthogonality")

        classifier_kwargs = args.get("classifier_config", {})
        if self._use_proxy_nca:
            classifier_kwargs["proxy_per_class"] = self._proxy_per_class
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True,
            attention_hook=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._lambda = args.get("base_lambda", 5)
        self._nb_negatives = args.get("nb_negatives", 2)
        self._margin = args.get("ranking_margin", 0.2)
        self._use_imprinted_weights = args.get("imprinted_weights", True)
        self._herding_indexes = []

        self._args = args
        self._args["_logs"] = {}

    def _after_task(self, inc_dataset):
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

        super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type == "knn":
            print("knn", self._evaluation_config)
            _, loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())
            features, targets = utils.extract_features(self._network, loader)

            return self._evaluate_knn(features, targets, test_loader)
        elif self._evaluation_type == "kmeans":
            print("kmeans + knn", self._evaluation_config)
            _, loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())
            features, targets = utils.extract_features(self._network, loader)

            means, means_targets = self._make_kmeans(
                features,
                targets,
            )
            return self._evaluate_knn(means, means_targets, test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for inputs, targets, _ in test_loader:
                ytrue.append(targets.numpy())

                inputs = inputs.to(self._device)
                logits = self._network(inputs)[1]
                if self._use_proxy_nca and self._proxy_per_class > 1:
                    logits = logits.view(-1, self._n_classes, self._proxy_per_class).sum(-1)

                if self._use_proxy_nca:
                    logits = -logits  # Proxy-nca outputs distances

                preds = logits.argmax(dim=1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _make_kmeans(self, features, targets):
        from sklearn.cluster import KMeans

        n_means = self._evaluation_config.get("n_means", 5)

        new_features = []
        new_targets = []
        for class_index in np.unique(targets):
            kmeans = KMeans(n_clusters=n_means)

            class_sample_indexes = np.where(targets == class_index)[0]
            class_features = features[class_sample_indexes]
            class_targets = np.ones((n_means,)) * class_index

            if self._evaluation_config.get("normalize", False):
                class_features = class_features / np.linalg.norm(class_features,
                                                                 axis=-1).reshape(-1, 1)

            kmeans.fit(class_features)
            new_features.append(kmeans.cluster_centers_)
            new_targets.append(class_targets)

        return np.concatenate(new_features), np.concatenate(new_targets)

    def _evaluate_knn(self, features, targets, test_loader):
        from sklearn.neighbors import KNeighborsClassifier

        if self._evaluation_config.get("n_neighbors") is None:
            n_neighbors = self._memory_per_class
        else:
            n_neighbors = self._evaluation_config.get("n_neighbors")

        if self._evaluation_config.get("normalize", False):
            features = features / np.linalg.norm(features, axis=-1).reshape(-1, 1)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=10)
        knn.fit(features, targets)

        features_test, targets_test = utils.extract_features(self._network, test_loader)

        if self._evaluation_config.get("normalize", False):
            features_test = features_test / np.linalg.norm(features_test, axis=-1).reshape(-1, 1)

        pred_targets = knn.predict(features_test)

        return pred_targets, targets_test

    def _before_task(self, train_loader, val_loader):
        if self._use_imprinted_weights:
            self._network.add_imprinted_classes(
                list(range(self._n_classes, self._n_classes + self._task_size)), self.inc_dataset
            )
        else:
            self._network.add_classes(self._task_size)
        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, *args, **kwargs):
        for p in self._network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        super()._train_task(*args, **kwargs)

    def _compute_loss(self, inputs, features_logits, targets, onehot_targets, memory_flags):
        features, logits, atts = features_logits
        if self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits, old_atts = self._old_model(inputs)

        if self._lambda_schedule:
            scheduled_lambda = self._lambda * math.sqrt(self._n_classes / self._task_size)
        else:
            scheduled_lambda = 1.

        # Classification loss is cosine + learned factor + softmax:
        if self._use_ams_ce:
            loss = losses.additive_margin_softmax_ce(logits, targets)
            self._metrics["ams"] += loss.item()
        elif self._use_npair:
            loss = losses.n_pair_loss(logits, targets)
            self._metrics["npair"] += loss.item()
        elif self._use_proxy_nca:
            #loss = losses.proxy_nca(logits, targets, self._n_classes,
            #                        self._proxy_per_class)
            loss = losses.proxy_nca_github(logits, targets, self._n_classes)
            self._metrics["nca"] += loss.item()
        else:
            if self._use_teacher_confidence and self._old_model is not None:
                loss = losses.cross_entropy_teacher_confidence(
                    self._network.post_process(logits), targets, F.softmax(old_logits, dim=1),
                    memory_flags
                )
                self._metrics["clf_conf"] += loss.item()
            else:
                loss = F.cross_entropy(self._network.post_process(logits), targets)
                self._metrics["clf"] += loss.item()

        if self._weights_orthogonality is not None:
            margin = self._weights_orthogonality.get("margin")
            ortho_loss = losses.weights_orthogonality(
                self._network.classifier.weights, margin=margin
            )
            loss += ortho_loss
            self._metrics["ortho"] += ortho_loss.item()

        if self._use_gor_reg:
            gor_loss = losses.global_orthogonal_regularization(features, targets)
            self._metrics["gor"] += gor_loss.item()
            loss += gor_loss

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

            if self._use_ranking:
                ranking_loss = losses.ucir_ranking(
                    logits,
                    targets,
                    self._n_classes,
                    self._task_size,
                    nb_negatives=self._nb_negatives,
                    margin=self._margin
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

        return loss
