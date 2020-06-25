import functools
import math

import torch
from torch.nn import functional as F

from inclearn.lib import vizualization


def mer_loss(new_logits, old_logits):
    """Distillation loss that is less important if the new model is unconfident.

    Reference:
        * Kim et al.
          Incremental Learning with Maximum Entropy Regularization: Rethinking
          Forgetting and Intransigence.

    :param new_logits: Logits from the new (student) model.
    :param old_logits: Logits from the old (teacher) model.
    :return: A float scalar loss.
    """
    new_probs = F.softmax(new_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)

    return torch.mean(((new_probs - old_probs) * torch.log(new_probs)).sum(-1), dim=0)


def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def spatial_pyramid_pooling(
    list_attentions_a,
    list_attentions_b,
    levels=[1, 2],
    pool_type="avg",
    weight_by_level=True,
    normalize=True,
    **kwargs
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        for j, level in enumerate(levels):
            if level > a.shape[2]:
                raise ValueError(
                    "Level {} is too big for spatial dim ({}, {}).".format(
                        level, a.shape[2], a.shape[3]
                    )
                )
            kernel_size = level // level

            if pool_type == "avg":
                a_pooled = F.avg_pool2d(a, (kernel_size, kernel_size))
                b_pooled = F.avg_pool2d(b, (kernel_size, kernel_size))
            elif pool_type == "max":
                a_pooled = F.max_pool2d(a, (kernel_size, kernel_size))
                b_pooled = F.max_pool2d(b, (kernel_size, kernel_size))
            else:
                raise ValueError("Invalid pool type {}.".format(pool_type))

            a_features = a_pooled.view(a.shape[0], -1)
            b_features = b_pooled.view(b.shape[0], -1)

            if normalize:
                a_features = F.normalize(a_features, dim=-1)
                b_features = F.normalize(b_features, dim=-1)

            level_loss = torch.frobenius_norm(a_features - b_features, dim=-1).mean(0)
            if weight_by_level:  # Give less importance for smaller cells.
                level_loss *= 1 / 2**j

            loss += level_loss

    return loss


def relative_teacher_distances(features_a, features_b, normalize=False, distance="l2", **kwargs):
    """Distillation loss between the teacher and the student comparing distances
    instead of embeddings.

    Reference:
        * Lu Yu et al.
          Learning Metrics from Teachers: Compact Networks for Image Embedding.
          CVPR 2019.

    :param features_a: ConvNet features of a model.
    :param features_b: ConvNet features of a model.
    :return: A float scalar loss.
    """
    if normalize:
        features_a = F.normalize(features_a, dim=-1, p=2)
        features_b = F.normalize(features_b, dim=-1, p=2)

    if distance == "l2":
        p = 2
    elif distance == "l1":
        p = 1
    else:
        raise ValueError("Invalid distance for relative teacher {}.".format(distance))

    pairwise_distances_a = torch.pdist(features_a, p=p)
    pairwise_distances_b = torch.pdist(features_b, p=p)

    return torch.mean(torch.abs(pairwise_distances_a - pairwise_distances_b))


def perceptual_features_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        # a of shape (b, c, w, h) to (b, c * w * h)
        a = a.view(bs, -1)
        b = b.view(bs, -1)

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        layer_loss = (F.pairwise_distance(a, b, p=2)**2) / (c * w * h)
        loss += torch.mean(layer_loss)

    return factor * (loss / len(list_attentions_a))


def perceptual_style_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        a = a.view(bs, c, w * h)
        b = b.view(bs, c, w * h)

        gram_a = torch.bmm(a, a.transpose(2, 1)) / (c * w * h)
        gram_b = torch.bmm(b, b.transpose(2, 1)) / (c * w * h)

        layer_loss = torch.frobenius_norm(gram_a - gram_b, dim=(1, 2))**2
        loss += layer_loss.mean()

    return factor * (loss / len(list_attentions_a))


def gradcam_distillation(gradients_a, gradients_b, activations_a, activations_b, factor=1):
    """Distillation loss between gradcam-generated attentions of two models.

    References:
        * Dhar et al.
          Learning without Memorizing
          CVPR 2019

    :param base_logits: [description]
    :param list_attentions_a: [description]
    :param list_attentions_b: [description]
    :param factor: [description], defaults to 1
    :return: [description]
    """
    attentions_a = _compute_gradcam_attention(gradients_a, activations_a)
    attentions_b = _compute_gradcam_attention(gradients_b, activations_b)

    assert len(attentions_a.shape) == len(attentions_b.shape) == 4
    assert attentions_a.shape == attentions_b.shape

    batch_size = attentions_a.shape[0]

    flat_attention_a = F.normalize(attentions_a.view(batch_size, -1), p=2, dim=-1)
    flat_attention_b = F.normalize(attentions_b.view(batch_size, -1), p=2, dim=-1)

    distances = torch.abs(flat_attention_a - flat_attention_b).sum(-1)

    return factor * torch.mean(distances)


def _compute_gradcam_attention(gradients, activations):
    alpha = F.adaptive_avg_pool2d(gradients, (1, 1))
    return F.relu(alpha * activations)


def mmd(x, y, sigmas=[1, 5, 10], normalize=False):
    """Maximum Mean Discrepancy with several Gaussian kernels."""
    # Flatten:
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    if len(sigmas) == 0:
        mean_dist = torch.mean(torch.pow(torch.pairwise_distance(x, y, p=2), 2))
        factors = (-1 / (2 * mean_dist)).view(1, 1, 1)
    else:
        factors = _get_mmd_factor(sigmas, x.device)

    if normalize:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

    xx = torch.pairwise_distance(x, x, p=2)**2
    yy = torch.pairwise_distance(y, y, p=2)**2
    xy = torch.pairwise_distance(x, y, p=2)**2

    k_xx, k_yy, k_xy = 0, 0, 0

    div = 1 / (x.shape[1]**2)

    k_xx = div * torch.exp(factors * xx).sum(0).squeeze()
    k_yy = div * torch.exp(factors * yy).sum(0).squeeze()
    k_xy = div * torch.exp(factors * xy).sum(0).squeeze()

    mmd_sq = torch.sum(k_xx) - 2 * torch.sum(k_xy) + torch.sum(k_yy)
    return torch.sqrt(mmd_sq)


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_factor(sigmas, device):
    sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
    sigmas = -1 / (2 * sigmas)
    return sigmas


def similarity_per_class(
    features,
    targets,
    goal_features,
    goal_targets,
    epoch,
    epochs,
    memory_flags,
    old_centroids_features=None,
    old_centroids_targets=None,
    factor=1.,
    scheduled=False,
    apply_centroids=True,
    initial_centroids=False
):
    loss = 0.
    counter = 0

    # We only keep new classes, no classes stored in memory
    indexes = ~memory_flags.bool()
    features = features[indexes]
    targets = targets[indexes].to(features.device)

    for target in torch.unique(targets):
        sub_features = features[targets == target]

        sub_goal_features = goal_features[goal_targets == target]
        if apply_centroids:
            sub_goal_features = sub_goal_features.mean(dim=0, keepdims=True)

        # We want the new real features to be similar to their old alter-ego ghosts:
        similarities = torch.mm(
            F.normalize(sub_features, dim=1, p=2),
            F.normalize(sub_goal_features, dim=1, p=2).T
        )
        loss += torch.clamp((1 - similarities).sum(), min=0.)
        counter += len(sub_features)

        if initial_centroids:
            # But we also want that the new real features stay close to what the
            # trained ConvNet though was best as first initialization:
            sub_centroids = old_centroids_features[old_centroids_targets == target]
            similarities = torch.mm(
                F.normalize(sub_features, dim=1, p=2), F.normalize(sub_centroids.T, dim=1, p=2)
            )
            loss += torch.clamp((1 - similarities).sum(), min=0.)
            counter += len(sub_features)

    if counter == 0:
        return 0.
    loss = factor * (loss / counter)

    if scheduled:
        loss = (1 - epoch / epochs) * loss

    if loss < 0.:
        raise ValueError(f"Negative loss value for PLC! (epoch={epoch}, epochs={epochs})")

    return loss


def semantic_drift_compensation(old_features, new_features, targets, sigma=0.2):
    """Returns SDC drift.

    # References:
        * Semantic Drift Compensation for Class-Incremental Learning
          Lu Yu et al.
          CVPR 2020
    """
    assert len(old_features) == len(new_features)

    with torch.no_grad():
        delta = new_features - old_features

        denominator = 1 / (2 * sigma**2)

        drift = torch.zeros(new_features.shape[1]).float().to(new_features.device)
        summed_w = 0.
        for target in torch.unique(targets):
            indexes = target == targets
            old_features_class = old_features[indexes]

            # Computing w, aka a weighting measuring how much an example
            # is representative based on its distance to the class mean.
            numerator = old_features_class - old_features_class.mean(dim=0)
            numerator = torch.pow(torch.norm(numerator, dim=1), 2)
            w = torch.exp(-numerator / denominator)

            tmp = (w[..., None] * delta[indexes])
            drift = drift + tmp.sum(dim=0)
            summed_w = summed_w + w.sum()
        drift = drift / summed_w

    return drift
