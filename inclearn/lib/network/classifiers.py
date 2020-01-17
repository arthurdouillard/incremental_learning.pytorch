import copy
import logging

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from inclearn.lib import distance as distance_lib
from inclearn.lib import utils

from .postprocessors import FactorScalar, HeatedUpScalar

logger = logging.getLogger(__name__)


class Classifier(nn.Module):

    def __init__(
        self, features_dim, *, use_bias, use_multi_fc=False, init="kaiming", device, **kwargs
    ):
        super().__init__()

        self.features_dim = features_dim
        self.use_bias = use_bias
        self.use_multi_fc = use_multi_fc
        self.init = init
        self.device = device

        self.n_classes = 0

        self.classifier = None

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, features):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.use_multi_fc:
            logits = []
            for classifier in self.classifier:
                logits.append(classifier(features))
            logits = torch.cat(logits, 1)
        else:
            logits = self.classifier(features)

        return logits, None

    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = nn.ModuleList([])

        new_classifier = self._gen_classifier(n_classes)
        self.classifier.append(new_classifier)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
        classifier = nn.Linear(self.features_dim, n_classes, bias=self.use_bias).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier


class CosineClassifier(nn.Module):

    def __init__(
        self,
        features_dim,
        device,
        *,
        proxy_per_class=1,
        distance="cosine",
        merging="softmax",
        scaling=1.,
        gamma=1.,
        use_bias=False,
        type=None,
    ):
        super().__init__()

        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma

        if isinstance(scaling, int):
            self.scaling = scaling
        else:
            self.scaling = FactorScalar(1.)

        if proxy_per_class > 1:
            logger.info("Using {} proxies per class.".format(proxy_per_class))

        self._task_idx = 0

    def on_task_end(self):
        self._task_idx += 1
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_epoch_end()

    def forward(self, features):
        if self.distance == "cosine":
            raw_similarities = distance_lib.cosine_similarity(features, self.weights)
        elif self.distance == "stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(self.weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(self.weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(self.weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

        return similarities, raw_similarities

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        assert similarities.shape[1] == self.n_classes * self.proxy_per_class

        if self.merging == "mean":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(-1, self.n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

    # ------------------
    # Weights management
    # ------------------

    @property
    def weights(self):
        return torch.cat([clf for clf in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(self.proxy_per_class * n_classes, self.features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        self._weights.append(new_weights)

        self.to(self.device)
        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
        self, class_indexes, inc_dataset, network, multi_class_diff="normal", type=None
    ):
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)
            else:
                if multi_class_diff == "normal":
                    std = torch.std(features_normalized, dim=0)
                    for _ in range(self.proxy_per_class):
                        new_weights.append(torch.normal(class_embeddings, std) * avg_weights_norm)
                elif multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.numpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(torch.tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".
                        format(multi_class_diff)
                    )

        new_weights = torch.stack(new_weights)
        self._weights.append(nn.Parameter(new_weights))

        self.to(self.device)
        self.n_classes += len(class_indexes)

        return self


class ProxyNCA(CosineClassifier):
    __slots__ = ("weights",)

    def __init__(
        self,
        *args,
        use_scaling=False,
        pre_relu=False,
        linear_end_relu=False,
        linear=False,
        mulfactor=3,
        gamma=1,
        merging="mean",
        metric="custom_distance",
        square_dist=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if use_scaling is True:
            self._scaling = FactorScalar(1.)
        elif use_scaling == "heatedup":
            self._scaling = HeatedUpScalar(16, 4, 6)
        elif isinstance(use_scaling, float):
            self._scaling = lambda x: use_scaling * x
        else:
            self._scaling = lambda x: x

        self.use_bias = False
        if linear:
            self.linear = nn.Sequential(
                nn.BatchNorm1d(self.features_dim), nn.ReLU(inplace=True),
                nn.Linear(self.features_dim, linear)
            )
            self.features_dim = linear
        else:
            self.linear = lambda x: x

        self.square_dist = square_dist
        self.gamma = gamma
        self.metric = metric
        print(metric)
        self.merging = merging
        self.mulfactor = mulfactor
        self.linear_end_relu = linear_end_relu
        self.pre_relu = pre_relu
        print("Proxy nca")

        self.weights = None

    def on_task_end(self):
        super().on_task_end()
        if isinstance(self._scaling, nn.Module):
            self._scaling.on_task_end()

    def forward(self, features):
        if isinstance(self.pre_relu, float):
            features = F.leaky_relu(features, self.pre_relu)
        elif self.pre_relu:
            features = F.relu(features)
        features = self.linear(features)
        if self.linear_end_relu:
            features = F.relu(features)

        P = self.weights
        P = self.mulfactor * F.normalize(P, p=2, dim=-1)
        X = self.mulfactor * F.normalize(features, p=2, dim=-1)

        if self.metric == "custom_distance":
            D = self.pairwise_distance(torch.cat(
                [X, P]
            ), squared=self.square_dist)[:X.size()[0], X.size()[0]:]
        elif self.metric == "neg_custom_distance":
            D = -self.pairwise_distance(torch.cat(
                [X, P]
            ), squared=self.square_dist)[:X.size()[0], X.size()[0]:]
        elif self.metric == "distance":
            D = torch.cdist(X, P)**2
        elif self.metric == "cosine":
            D = torch.mm(X, P.t())
        else:
            raise ValueError("Unknown metric {}.".format(self.metric))

        if self.proxy_per_class > 1:
            D = self._reduce_proxies(D)

        return self._scaling(D)

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        assert similarities.shape[1] == self.n_classes * self.proxy_per_class

        if self.merging == "mean":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(-1, self.n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

    @staticmethod
    def pairwise_distance(a, squared=False):
        """Computes the pairwise distance matrix with numerical stability."""
        pairwise_distances_squared = torch.add(
            a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
            torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
        ) - 2 * (torch.mm(a, torch.t(a)))

        # Deal with numerical inaccuracies. Set small negatives to zero.
        pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

        # Get the mask where the zero distances are at.
        error_mask = torch.le(pairwise_distances_squared, 0.0)

        # Optionally take the sqrt.
        if squared:
            pairwise_distances = pairwise_distances_squared
        else:
            pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

        # Undo conditionally adding 1e-16.
        pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

        # Explicitly set diagonals to zero.
        mask_offdiagonals = 1 - torch.eye(
            *pairwise_distances.size(), device=pairwise_distances.device
        )
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

        return pairwise_distances

    def add_custom_weights(self, weights):
        weights = torch.tensor(weights)

        if self.weights is not None:
            placeholder = nn.Parameter(
                torch.zeros(self.weights.shape[0] + weights.shape[0], self.features_dim)
            )
            placeholder.data[:self.weights.shape[0]] = copy.deepcopy(self.weights.data)
            placeholder.data[self.weights.shape[0]:] = weights

            self.weights = placeholder
        else:
            self.weights = weights

        self.to(self.device)

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(
            torch.zeros(self.proxy_per_class * (self.n_classes + n_classes), self.features_dim)
        )
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        if self.weights is not None:
            new_weights.data[:self.n_classes *
                             self.proxy_per_class] = copy.deepcopy(self.weights.data)

        del self.weights
        self.weights = new_weights

        if self.use_bias:
            new_bias = nn.Parameter(
                torch.zeros(self.proxy_per_class * (self.n_classes + n_classes))
            )
            nn.init.constant_(new_bias, 0.1)
            if self.bias is not None:
                new_bias.data[:self.n_classes *
                              self.proxy_per_class] = copy.deepcopy(self.bias.data)

            del self.bias
            self.bias = new_bias

        self.to(self.device)
        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
        self,
        class_indexes,
        inc_dataset,
        network,
        use_weights_norm=True,
        multi_class_diff="normal",
        **kwargs
    ):
        if self.proxy_per_class > 1:
            print("Multi class diff {}.".format(multi_class_diff))

        # We are assuming the class indexes are contiguous!
        n_classes = self.n_classes
        self.add_classes(len(class_indexes))
        if n_classes == 0:
            return

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()
        if not use_weights_norm:
            print("Not using avg weight norm")
            avg_weights_norm = torch.ones_like(avg_weights_norm)

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)
            else:
                if multi_class_diff == "normal":
                    std = torch.std(features_normalized, dim=0)
                    for _ in range(self.proxy_per_class):
                        new_weights.append(torch.normal(class_embeddings, std) * avg_weights_norm)
                elif multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.numpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(torch.tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".
                        format(multi_class_diff)
                    )

        new_weights = torch.stack(new_weights)
        self.weights.data[-new_weights.shape[0]:] = new_weights.to(self.device)

        return self
