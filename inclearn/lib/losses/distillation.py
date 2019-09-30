import torch
from torch.nn import functional as F


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


def residual_attention_distillation(list_attentions_a, list_attentions_b, use_depth_weights=True):
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

    depth_weights = torch.tensor(list(range(len(list_attentions_a), -1, -1))) + 1
    depth_weights = F.normalize(depth_weights.float(), dim=0).to(list_attentions_a[0].device)

    loss = 0.
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
        b = b.sum(dim=1).view(b.shape[0], -1)

        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.frobenius_norm(a - b)
        if use_depth_weights:
            layer_loss = depth_weights[i] * layer_loss
        loss += layer_loss

    loss = 2 * loss

    if use_depth_weights:
        return loss
    return loss / len(list_attentions_a)


def relative_teacher_distances(features_a, features_b, normalize_features=False):
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
    if normalize_features:
        features_a = F.normalize(features_a, dim=-1, p=2)
        features_b = F.normalize(features_b, dim=-1, p=2)

    pairwise_distances_a = torch.pdist(features_a, p=2)
    pairwise_distances_b = torch.pdist(features_b, p=2)

    return torch.mean(torch.abs(pairwise_distances_a - pairwise_distances_b))
