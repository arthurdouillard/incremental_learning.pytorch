import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import utils


def weights_orthogonality(weights, margin=0.):
    """Regularization forcing the weights to be disimilar.

    :param weights: Learned parameters of shape (n_classes, n_features).
    :param margin: Margin to force even more the orthogonality.
    :return: A float scalar loss.
    """
    normalized_weights = F.normalize(weights, dim=1, p=2)
    similarities = torch.mm(normalized_weights, normalized_weights.t())

    # We are ignoring the diagonal made of identity similarities:
    similarities = similarities[torch.eye(similarities.shape[0]) == 0]

    return torch.mean(F.relu(similarities + margin))


def ortho_reg(weights, config):
    """Regularization forcing the weights to be orthogonal without removing negative
    correlation.

    Reference:
        * Regularizing CNNs with Locally Constrained Decorrelations
          Pau et al.
          ICLR 2017

    :param weights: Learned parameters of shape (n_classes, n_features).
    :return: A float scalar loss.
    """
    normalized_weights = F.normalize(weights, dim=1, p=2)
    similarities = torch.mm(normalized_weights, normalized_weights.t())

    # We are ignoring the diagonal made of identity similarities:
    similarities = similarities[torch.eye(similarities.shape[0]) == 0]

    x = config.get("lambda", 10.) * (similarities - 1.)

    return config.get("factor", 1.) * torch.log(1 + torch.exp(x)).sum()


def global_orthogonal_regularization(
    features, targets, factor=1., normalize=False, sampling="per_target"
):
    """Global Orthogonal Regularization (GOR) forces features of different
    classes to be orthogonal.

    # Reference:
        * Learning Spread-out Local Feature Descriptors.
          Zhang et al.
          ICCV 2016.

    :param features: A flattened extracted features.
    :param targets: Sparse targets.
    :return: A float scalar loss.
    """
    if normalize:
        features = F.normalize(features, dim=1, p=2)

    positive_indexes, negative_indexes = [], []
    targets = targets.cpu().numpy()
    unique_targets = set(targets)
    if len(unique_targets) == 0:
        return torch.tensor(0.)

    if sampling == "per_target":
        for target in unique_targets:
            positive_index = np.random.choice(np.where(targets == target)[0], 1)
            negative_index = np.random.choice(np.where(targets != target)[0], 1)

            positive_indexes.append(positive_index)
            negative_indexes.append(negative_index)
    elif sampling == "per_sample":
        for positive_index, target in enumerate(targets):
            negative_index = np.random.choice(np.where(targets != target)[0], 1)

            positive_indexes.append(positive_index)
            negative_indexes.append(negative_index)
    elif sampling == "full":
        pair_indexes = set()
        for positive_index, target in enumerate(targets):
            for negative_index in np.where(targets != target)[0]:
                pair_indexes.add(tuple(sorted((positive_index, negative_index))))

        for p, n in pair_indexes:
            positive_indexes.append(p)
            negative_indexes.append(n)
    else:
        raise ValueError("Unknown sampling type {}.".format(sampling))

    positive_indexes = torch.LongTensor(positive_indexes)
    negative_indexes = torch.LongTensor(negative_indexes)

    positive_features = features[positive_indexes]
    negative_features = features[negative_indexes]

    similarities = torch.sum(torch.mul(positive_features, negative_features), 1)
    features_dim = features.shape[1]

    first_moment = torch.mean(similarities)
    second_moment = torch.mean(torch.pow(similarities, 2))

    loss = torch.pow(first_moment, 2) + torch.clamp(second_moment - 1. / features_dim, min=0.)

    return factor * loss


def double_soft_orthoreg(weights, config):
    """Extention of the Soft Ortogonality reg, forces the Gram matrix of the
    weight matrix to be close to identity.

    Also called DSO.

    References:
        * Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
          Bansal et al.
          NeurIPS 2018

    :param weights: Learned parameters of shape (n_classes, n_features).
    :return: A float scalar loss.
    """
    wTw = torch.mm(weights.t(), weights)
    so_1 = torch.frobenius_norm(wTw - torch.eye(wTw.shape[0]).to(weights.device))

    wwT = torch.mm(weights, weights.t())
    so_2 = torch.frobenius_norm(wwT - torch.eye(wwT.shape[0]).to(weights.device))

    if config["squared"]:
        so_1 = torch.pow(so_1, 2)
        so_2 = torch.pow(so_2, 2)

    return config["factor"] * (so_1 + so_2)


def mutual_coherence_regularization(weights, config):
    """Forces weights orthogonality by reducing the highest correlation between
    the weights.

    Also called MC.

    References:
        * Compressed sensing
          David L Donoho.
          Transactions on information theory 2016

    :param weights: Learned parameters of shape (n_classes, n_features).
    :return: A float scalar loss.
    """
    wTw = torch.mm(weights.t(), weights)
    x = wTw - torch.eye(wTw.shape[0]).to(weights.device)

    loss = utils.matrix_infinity_norm(x)

    return config["factor"] * loss


def spectral_restricted_isometry_property_regularization(weights, config):
    """Requires that every set of columns of the weights, with cardinality no
    larger than k, shall behave like an orthogonal system.

    Also called SRIP.

    References:
        * Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
          Bansal et al.
          NeurIPS 2018

    :param weights: Learned parameters of shape (n_classes, n_features).
    :return: A float scalar loss.
    """
    wTw = torch.mm(weights.t(), weights)
    x = wTw - torch.eye(wTw.shape[0]).to(weights.device)

    _, s, _ = torch.svd(x)

    loss = s[0]
    return config["factor"] * loss
