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


def proxy_nca_github(D, targets, nb_classes, **config):
    #return proxy_nca(-D, targets, nb_classes, 1)

    T = binarize_and_smooth_labels(T=targets, nb_classes=nb_classes,
                                   smoothing_const=0.).to(targets.device)

    # cross entropy with distances as logits, one hot labels
    # note that compared to proxy nca, positive not excluded in denominator
    #loss = torch.sum(-T * F.log_softmax(D, -1), -1)
    loss = additive_margin_softmax_ce(D, targets, **config)
    return loss

    return loss.mean()


def proxy_nca(similarities, targets, nb_classes, proxy_per_class):
    """NCA with proxies.

    Reference:
        * No Fuss Distance Metric Learning using Proxies.
          Movshovitz-Attias et al.
          AAAI 2017.

    :param similarities: A batch of similarities (or negative distance) between
                         inputs & proxies.
    :param targets: Sparse targets.
    :param nb_classes: Number of classes.
    :param proxy_per_class: Number of proxy per class.
    :return: A float scalar loss.
    """
    assert similarities.shape[1] == (nb_classes * proxy_per_class)

    positive_proxies_mask = torch.zeros_like(similarities, dtype=torch.bool)
    indexes = torch.arange(similarities.shape[0])
    for i in range(proxy_per_class):
        positive_proxies_mask[indexes, i + proxy_per_class * targets] = True

    similar_pos = similarities[positive_proxies_mask].view(similarities.shape[0], proxy_per_class)
    most_similar_pos, _ = similar_pos.max(dim=-1)

    negative_proxies = ~positive_proxies_mask
    negative_similarities = similarities[negative_proxies]
    negative_similarities = negative_similarities.view(
        similarities.shape[0], proxy_per_class * (nb_classes - 1)
    )

    denominator = torch.exp(negative_similarities).sum(-1)
    numerator = most_similar_pos

    denominator = torch.exp(similarities).sum(-1)
    loss = -torch.mean(numerator - torch.log(denominator))
    if loss < 0:
        import pdb
        pdb.set_trace()
    return loss


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


def additive_margin_softmax_ce(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.,
    exclude_pos_denominator=False,
    hinge_proxynca=False,
    memory_flags=None,
    old_classes_temperature=None,
    new_classes_lambda=None,
    update_only_pos=False
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :param old_classes_temperature: Divide all old classes similarities by this constant.
    :param new_classes_lambda: Multiply all new classes similarities that are in the
                               numerator by this constant.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if old_classes_temperature:
        temps = torch.ones_like(similarities)
        temps[memory_flags.bool()] = old_classes_temperature
        similarities.div_(temps)

    if exclude_pos_denominator:
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        if new_classes_lambda is not None and memory_flags is not None:
            lambdas = torch.ones_like(denominator)
            lambdas[~memory_flags.bool()] = new_classes_lambda
            denominator.mul_(lambdas)

        if update_only_pos:
            denominator = denominator.detach()

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
    return github_ucir_ranking_mr(logits, targets, n_classes, task_size, nb_negatives, margin)
    # Ranking loss maximizing the inter-class separation between old & new:

    # 1. Fetching from the batch only samples from the batch that belongs
    #    to old classes:
    old_indexes = targets.lt(n_classes - 1)
    old_logits = logits[old_indexes]
    old_targets = targets[old_indexes]

    # 2. Getting positive values, aka ground-truth's logit predictions:
    old_values = old_logits[torch.arange(len(old_logits)), old_targets]
    old_values = old_values.repeat(nb_negatives, 1).t().contiguous().view(-1)

    # 3. Getting top-k negative values:
    nb_old_classes = n_classes - task_size
    negative_indexes = old_logits[..., nb_old_classes:].argsort(dim=1, descending=True)[
        ..., :nb_negatives] + nb_old_classes
    new_values = old_logits[torch.arange(len(old_logits)).view(-1, 1), negative_indexes].view(-1)

    return F.margin_ranking_loss(
        old_values, new_values, -torch.ones(len(old_values)).to(logits.device), margin=margin
    )


def github_ucir_ranking_mr(logits, targets, n_classes, task_size, nb_negatives=2, margin=0.2):
    gt_index = torch.zeros(logits.size()).to(logits.device)
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
    gt_scores = logits.masked_select(gt_index)
    #get top-K scores on novel classes
    num_old_classes = logits.shape[1] - task_size
    max_novel_scores = logits[:, num_old_classes:].topk(nb_negatives, dim=1)[0]
    #the index of hard samples, i.e., samples of old classes
    hard_index = targets.lt(num_old_classes)
    hard_num = torch.nonzero(hard_index).size(0)
    #print("hard examples size: ", hard_num)
    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        #print("hard example gt scores: ", gt_scores.size(), gt_scores)
        #print("hard example max novel scores: ", max_novel_scores.size(), max_novel_scores)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
            max_novel_scores.view(-1, 1), torch.ones(hard_num*nb_negatives).to(logits.device))
        return loss
    return torch.tensor(0).float()


def n_pair_loss(logits, targets):
    return NPairLoss()(F.relu(logits), targets)
    #return NPairAngularLoss(angle_bound=math.tan(50)**2)(F.relu(logits), targets)
    #return apply_loss(logits, targets)

    batch_similarities = torch.mm(logits, logits.t())

    targets_col = targets.view(targets.shape[0], 1)
    targets_mat = (targets_col == targets_col.t()).float()
    targets_mat_normalized = targets_mat / (targets_mat.sum(dim=-1).float() - 1.)

    mask = torch.eye(targets.shape[0]).byte()
    batch_similarities_nodiag = batch_similarities[~mask].view(mask.shape[0], mask.shape[0] - 1)
    targets_mat_normalized_nodiag = targets_mat_normalized[~mask].view(
        mask.shape[0], mask.shape[0] - 1
    )

    return F.binary_cross_entropy(
        F.softmax(batch_similarities_nodiag, dim=-1), targets_mat_normalized_nodiag
    )


def apply_loss(embeddings, targets, loss_type="npair"):

    def f(x):
        return x.view(1, -1)

    def npair(anchor, positive, negative):
        an = torch.mm(f(anchor), f(negative).t())[0][0]
        ap = torch.mm(f(anchor), f(positive).t())[0][0]
        return an - ap

    def angular(anchor, positive, negative):
        alpha = 40
        cst = torch.pow(torch.tan(alpha), 2)

        ap_n = torch.mm(f(anchor + positive), f(negative).t())
        ap = torch.mm(f(anchor), f(positive).t())

        return 4 * cst * ap_n - 2 * (1 + cst) * ap

    loss = 0.
    for anchor_index, target_a in enumerate(targets):  # For X_anchor in batch:
        anchor_loss = 0.

        for positive_index in targets.eq(target_a):
            if anchor_index == positive_index:
                continue

            for negative_index, target_n in enumerate(targets):
                if target_a == target_n:
                    continue

            if loss_type == "npair":
                _loss = npair(
                    embeddings[anchor_index], embeddings[positive_index.item()],
                    embeddings[negative_index]
                )
            else:
                raise ValueError("bad loss ", loss_type)

            anchor_loss += torch.exp(_loss)

        loss += torch.log(1 + anchor_loss)

    return loss / len(targets)


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
           # + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i + 1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1 + x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors**2 + positives**2) / anchors.shape[0]


class AngularLoss(NPairLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss


class NPairAngularLoss(AngularLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(NPairAngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.n_pair_angular_loss(anchors, positives, negatives, self.angle_bound) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    def n_pair_angular_loss(self, anchors, positives, negatives, angle_bound=1.):
        """
        Calculates N-Pair angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar, n-pair_loss + lambda * angular_loss
        """
        n_pair = self.n_pair_loss(anchors, positives, negatives)
        angular = self.angular_loss(anchors, positives, negatives, angle_bound)

        return (n_pair + self.lambda_ang * angular) / (1 + self.lambda_ang)
