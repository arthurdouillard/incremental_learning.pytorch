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


def residual_attention_distillation(
    list_attentions_a,
    list_attentions_b,
    use_depth_weights=False,
    collapse_channels="channels",
    preprocess="square",
    normalize=True,
    memory_flags=None,
    only_old=False,
    aggreg=False,
    layer_weights=None,
    weight_per_channels=False,
    task_percent=None,
    percent_mult=None,
    min_inverse_percent=5,
    **kwargs
):
    """Residual attention distillation between several attention maps between
    a teacher and a student network.

    Reference:
        * S. Zagoruyko and N. Komodakis.
          Paying more attention to attention: Improving the performance of
          convolutional neural networks via attention transfer.
          ICLR 2016.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    if use_depth_weights is True:
        depth_weights = torch.tensor(list(range(len(list_attentions_a), -1, -1))) + 1
        depth_weights = F.normalize(depth_weights.float(), dim=0).to(list_attentions_a[0].device)
    if isinstance(use_depth_weights, str) and use_depth_weights == "reverse":
        depth_weights = torch.tensor(list(range(len(list_attentions_a)))) + 1
        depth_weights = F.normalize(depth_weights.float(), dim=0).to(list_attentions_a[0].device)

    if layer_weights is not None and len(layer_weights) != len(list_attentions_a):
        raise ValueError(
            "Number of weights & number of activations not matching"
            " ({} vs {}).".format(len(layer_weights), len(list_attentions_a))
        )

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        if preprocess == "square":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif preprocess == "relu_square":
            a = torch.pow(F.relu(a), 2)
            b = torch.pow(F.relu(b), 2)
        elif preprocess == "abs":
            a = torch.abs(a)
            b = torch.abs(b)
        elif preprocess is None:
            pass
        else:
            raise NotImplementedError(
                "Unknown preprocess for residual attention: {}.".format(preprocess)
            )

        if isinstance(collapse_channels, int):
            scores_a = F.adaptive_avg_pool2d(a, (1, 1)).view(a.shape[0], a.shape[1])
            topk_channels = torch.argsort(scores_a, descending=True, dim=1)
            topk_channels = topk_channels[..., :collapse_channels]

            a = a[torch.arange(len(a)), topk_channels.t()].transpose(0, 1)
            b = b[torch.arange(len(b)), topk_channels.t()].transpose(0, 1)

            a = a[:, topk_channels]
            b = b[:, topk_channels]
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "percent":
            scores_a = F.adaptive_avg_pool2d(a, (1, 1)).view(
                a.shape[0], a.shape[1]
            )  # shape of (b, c)
            topk_channels = torch.argsort(scores_a, descending=True, dim=-1)

            mult = a.shape[1] * task_percent
            if percent_mult is not None:
                mult *= percent_mult[i]
            topk_channels = topk_channels[..., :math.ceil(mult)]

            # Horrible but only way to be fast.
            # Index for the whole batch the desired channels (can be multiple channels).
            a = a[torch.arange(len(a)), topk_channels.t()].transpose(0, 1)
            b = b[torch.arange(len(b)), topk_channels.t()].transpose(0, 1)

            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "inverse_percent":
            scores_a = F.adaptive_avg_pool2d(a, (1, 1)).view(
                a.shape[0], a.shape[1]
            )  # shape of (b, c)
            topk_channels = torch.argsort(scores_a, descending=True, dim=-1)

            mult = math.ceil(a.shape[1] * (1 - task_percent))
            mult = min(mult, min_inverse_percent)
            topk_channels = topk_channels[..., :mult]

            # Horrible but only way to be fast.
            # Index for the whole batch the desired channels (can be multiple channels).
            a = a[torch.arange(len(a)), topk_channels.t()].transpose(0, 1)
            b = b[torch.arange(len(b)), topk_channels.t()].transpose(0, 1)

            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "mean":
            a = a.mean(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.mean(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w])
            b = torch.cat([b_h, b_w])
        elif collapse_channels == "spatial_correlation":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)

            a = 1 / (a_h.shape[1] - 1) * torch.mm(a_h, a_w.T)
            b = 1 / (b_h.shape[1] - 1) * torch.mm(b_h, b_w.T)
        elif collapse_channels == "spatial_percent":
            scores_a = F.adaptive_avg_pool2d(a, (1, 1)).view(
                a.shape[0], a.shape[1]
            )  # shape of (b, c)
            topk_channels = torch.argsort(scores_a, descending=True, dim=-1)
            topk_channels = topk_channels[..., :math.ceil(a.shape[1] * task_percent)]

            # Horrible but only way to be fast.
            # Index for the whole batch the desired channels (can be multiple channels).
            aa = a[torch.arange(len(a)), topk_channels.t()].transpose(0, 1)
            bb = b[torch.arange(len(b)), topk_channels.t()].transpose(0, 1)

            try:
                a_h = aa.sum(dim=3).view(aa.shape[0], -1)
                b_h = bb.sum(dim=3).view(bb.shape[0], -1)
                a_w = aa.sum(dim=2).view(aa.shape[0], -1)
                b_w = bb.sum(dim=2).view(bb.shape[0], -1)
            except:
                import pdb
                pdb.set_trace()
            a = torch.cat([a_h, a_w])
            b = torch.cat([b_h, b_w])
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if not aggreg:
            layer_loss = torch.frobenius_norm(a - b)
        elif aggreg == "paper":
            layer_loss = (a - b).pow(2).mean()
        elif aggreg == "mean":
            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        elif aggreg == "secondmoment":
            dist = torch.frobenius_norm(a - b, dim=-1)
            first_moment = torch.mean(dist)
            second_moment = torch.mean(torch.pow(dist, 2))
            layer_loss = torch.pow(first_moment,
                                   2) + torch.clamp(second_moment - 1. / a.shape[-1], min=0.)
        elif isinstance(aggreg, list):
            # Maximum Mean Discrepancy, aggreg is a list of sigmas for Gaussian kernels
            layer_loss = mmd(a, b, sigmas=tuple(aggreg))
        else:
            raise NotImplementedError("Unknown aggreg method for RAD: {}.".format(aggreg))

        if use_depth_weights:
            layer_loss = depth_weights[i] * layer_loss
        elif layer_weights:
            layer_loss = layer_weights[i] * layer_loss
        loss += layer_loss

    if use_depth_weights or aggreg == "paper":
        return loss
    return loss / len(list_attentions_a)


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
