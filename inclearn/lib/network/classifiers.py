import copy

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from inclearn.lib import utils

from .postprocessors import ConstantScalar, FactorScalar, HeatedUpScalar


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

        return logits

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
        use_bias=False,
        proxy_per_class=1,
        bn_normalize=False,
        freeze_bn=False,
        type=None
    ):
        super().__init__()

        self.n_classes = 0
        self.weights = None
        self.bias = None
        self.use_bias = use_bias
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device

        if bn_normalize:
            print("Normalizing with BN.")
            self.bn = nn.BatchNorm1d(features_dim, affine=False)
        else:
            self.bn = None

        if proxy_per_class > 1:
            print("Using {} proxies per class.".format(proxy_per_class))

        self.freeze_bn = freeze_bn
        self._task_idx = 0

    def on_task_end(self):
        self._task_idx += 1

    def on_epoch_end(self):
        pass

    def forward(self, features):
        if self.bn:
            if self.freeze_bn and self._task_idx > 0 and self.bn.training:
                self.bn.eval()
            features_norm = self.bn(features)
            if self.use_bias:
                features_norm = features_norm
        else:
            features_norm = features / (features.norm(dim=1)[:, None] + 1e-8)

        weights_norm = self.weights / (self.weights.norm(dim=1)[:, None] + 1e-8)

        similarities = torch.mm(features_norm, weights_norm.transpose(0, 1))

        if self.use_bias:
            similarities += self.bias

        return similarities

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


class ProxyNCA(CosineClassifier):

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
            self._scaling = ConstantScalar(constant=use_scaling)
        else:
            self._scaling = ConstantScalar()

        if linear:
            self.linear = nn.Linear(64, 64)
        else:
            self.linear = ConstantScalar()

        self.square_dist = square_dist
        self.gamma = gamma
        self.metric = metric
        print(metric)
        self.merging = merging
        self.mulfactor = mulfactor
        self.linear_end_relu = linear_end_relu
        self.pre_relu = pre_relu
        print("Proxy nca")

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
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(-1, self.n_classes, self.proxy_per_class).max(-1)[0]
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


class SoftTriple(ProxyNCA):

    def __init__(self, *args, normalize=True, proxy_per_class=10, gamma=0.1, op="dot", **kwargs):
        super().__init__(*args, proxy_per_class=proxy_per_class, **kwargs)
        self.normalize = normalize
        self.gamma = gamma
        self.op = op

        assert op in ("dot", "distance"), op

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, dim=-1, p=2)
            w = F.normalize(self.weights, dim=-1, p=2)
        else:
            w = self.weights

        centers_per_class = w.view(self.n_classes, self.proxy_per_class, -1)

        # Merge centers of same class by soft max:
        merged_centers = []
        for c in range(self.n_classes):
            if self.op == "dot":
                xTw = torch.mm(x, centers_per_class[c].t())
            elif self.op == "distance":
                xTw = self.pairwise_distance(torch.cat([x, w]),
                                             squared=True)[:x.size()[0], x.size()[0]:]

            s_c = (F.softmax((1 / self.gamma) * xTw, dim=-1) * xTw).sum(-1)

            merged_centers.append(s_c)

        return torch.stack(merged_centers).transpose(1, 0)
