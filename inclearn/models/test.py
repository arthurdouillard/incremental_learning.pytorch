import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, losses, network, utils
from inclearn.models.icarl import ICarl


class Test(ICarl):
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

        self._use_distil = args.get("distillation_loss", True)
        self._lambda_schedule = args.get("lambda_schedule", True)
        self._use_ranking = args.get("ranking_loss", True)
        self._scaling_factor = args.get("scaling_factor", True)

        self._use_npair_distil = args.get("npair_distil", False)

        self._network = network.BasicNet(
            args["convnet"],
            device=self._device,
            classifier_type="intersimi",
            scaling_factor=self._scaling_factor,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._lambda = args.get("base_lambda", 5)
        self._nb_negatives = args.get("nb_negatives", 2)
        self._margin = args.get("ranking_margin", 0.2)
        self._use_imprinted_weights = args.get("imprinted_weights", True)

        self._herding_indexes = []

    def _before_task(self, train_loader, val_loader):
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
        features, logits = features_logits

        loss = losses.n_pair_loss(logits, targets)

        if self._use_distil and self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits = self._old_model(inputs)

            # Distillation loss fixing the deviation problem:
            if self._lambda_schedule:
                scheduled_lambda = self._lambda * math.sqrt(
                    self._n_classes / self._task_size
                )
            else:
                scheduled_lambda = 1.

            distil_loss = scheduled_lambda * F.cosine_embedding_loss(
                features, old_features,
                torch.ones(inputs.shape[0]).to(self._device)
            )

            loss += distil_loss

        return loss

    """
    def _after_task(self, inc_dataset):
        self.build_examplars(inc_dataset)

        self._old_model = self._network.copy().freeze()

    def build_examplars(self, inc_dataset):
        data_memory, targets_memory = [], []
        data_clusters, targets_clusters = [], []

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test")
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )

            indexes = np.random.randint(len(inputs), size=self._memory_per_class)

            data_memory.append(inputs[indexes])
            targets_memory.append(targets[indexes])

            n_clusters = 8
            clusters = self._select_clusters(
                np.concatenate((features, features_flipped)),
                n_clusters=n_clusters)
            data_clusters.append(clusters)
            targets_clusters.append(np.ones(n_clusters) * class_idx)

        self._data_clusters = np.concatenate(data_clusters)
        self._targets_clusters = np.concatenate(targets_clusters)

        self._data_memory = np.concatenate(data_memory)
        self._targets_memory = np.concatenate(targets_memory)

    def _select_clusters(self, features, n_clusters=8):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(features)

        return kmeans.cluster_centers_

    def _eval_task(self, loader):
        features, targets = utils.extract_features(self._network, loader)

        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(
            n_neighbors=8
        )
        knn.fit(self._data_clusters, self._targets_clusters)

        ypred = knn.predict(features)

        return ypred, targets
    """
