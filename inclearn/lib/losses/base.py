import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T)
    return T


def cross_entropy_teacher_confidence(similarities, targets, old_confidence, memory_indexes):
    memory_indexes = memory_indexes.byte()

    per_sample_losses = F.cross_entropy(similarities, targets, reduction="none")

    memory_losses = per_sample_losses[memory_indexes]
    new_losses = per_sample_losses[~memory_indexes]

    memory_old_confidence = old_confidence[memory_indexes]
    memory_targets = targets[memory_indexes]

    right_old_confidence = memory_old_confidence[torch.arange(memory_old_confidence.shape[0]),
                                                 memory_targets]
    hard_indexes = right_old_confidence.le(0.5)

    factors = 2 * (1 + (1 - right_old_confidence[hard_indexes]))

    loss = torch.mean(
        torch.cat(
            (new_losses, memory_losses[~hard_indexes], memory_losses[hard_indexes] * factors)
        )
    )

    return loss


def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )


def ucir_ranking(logits, targets, n_classes, task_size, nb_negatives=2, margin=0.2):
    """Hinge loss from UCIR.

    Taken from: https://github.com/hshustc/CVPR19_Incremental_Learning

    # References:
        * Learning a Unified Classifier Incrementally via Rebalancing
          Hou et al.
          CVPR 2019
    """
    gt_index = torch.zeros(logits.size()).to(logits.device)
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
    gt_scores = logits.masked_select(gt_index)
    # get top-K scores on novel classes
    num_old_classes = logits.shape[1] - task_size
    max_novel_scores = logits[:, num_old_classes:].topk(nb_negatives, dim=1)[0]
    # the index of hard samples, i.e., samples of old classes
    hard_index = targets.lt(num_old_classes)
    hard_num = torch.nonzero(hard_index).size(0)

    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
            max_novel_scores.view(-1, 1), torch.ones(hard_num*nb_negatives).to(logits.device))
        return loss

    return torch.tensor(0).float()
