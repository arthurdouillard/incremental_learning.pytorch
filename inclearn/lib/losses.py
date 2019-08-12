import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cross_entropy_teacher_confidence(similarities, targets, old_confidence, memory_indexes):
    memory_indexes = memory_indexes.byte()

    per_sample_losses = F.cross_entropy(similarities, targets, reduction="none")

    memory_losses = per_sample_losses[memory_indexes]
    new_losses = per_sample_losses[~memory_indexes]

    memory_old_confidence = old_confidence[memory_indexes]
    memory_targets = targets[memory_indexes]

    right_old_confidence = memory_old_confidence[torch.arange(memory_old_confidence.
                                                              shape[0]), memory_targets]
    hard_indexes = right_old_confidence.le(0.5)

    factors = 2 * (1 + (1 - right_old_confidence[hard_indexes]))

    loss = torch.mean(
        torch.cat(
            (new_losses, memory_losses[~hard_indexes], memory_losses[hard_indexes] * factors)
        )
    )

    return loss


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

    positive_proxies_mask = torch.zeros_like(similarities).byte()
    indexes = torch.arange(similarities.shape[0])
    for i in range(proxy_per_class):
        positive_proxies_mask[indexes, i + proxy_per_class * targets] = 1

    similar_pos = similarities[positive_proxies_mask].view(similarities.shape[0], proxy_per_class)
    most_similar_pos, _ = similar_pos.max(dim=-1)

    negative_proxies = ~positive_proxies_mask
    negative_similarities = similarities[negative_proxies]
    negative_similarities = negative_similarities.view(
        similarities.shape[0], proxy_per_class * (nb_classes - 1)
    )

    denominator = torch.exp(negative_similarities).sum(-1)
    numerator = torch.exp(most_similar_pos)

    return -torch.mean(torch.log(numerator / denominator))


def additive_margin_softmax_ce(similarities, targets, s=30, m=0.4):
    """Compute AMS cross-entropy loss.

    Reference:
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    Inspired by (but speed up x7):
        * https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param s: Multiplicative factor, can be learned.
    :param m: Margin applied on the "right" similarities.
    :return: A float scalar loss.
    """
    numerator = s * (torch.diagonal(similarities.transpose(0, 1)[targets]) - m)

    neg_denominator = torch.exp(s * similarities)
    mask = torch.ones(similarities.shape[0], similarities.shape[1]).to(similarities.device)
    mask[torch.arange(similarities.shape[0]), targets] = 0
    neg_denominator = neg_denominator * mask

    denominator = torch.exp(numerator) + torch.sum(neg_denominator, dim=1)
    loss = numerator - torch.log(denominator)
    return -torch.mean(loss)


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

        #layer_loss = torch.frobenius_norm(a - b)
        layer_loss = (1 - torch.mm(a, b.t())).sum()
        if use_depth_weights:
            layer_loss = depth_weights[i] * layer_loss
        loss += layer_loss

    loss = loss

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


def global_orthogonal_regularization(features, targets, normalize=False):
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
    for target in set(targets):
        positive_index = np.random.choice(np.where(targets == target)[0], 1)
        negative_index = np.random.choice(np.where(targets != target)[0], 1)

        positive_indexes.append(positive_index)
        negative_indexes.append(negative_index)

    positive_indexes = torch.LongTensor(positive_indexes)
    negative_indexes = torch.LongTensor(negative_indexes)

    positive_features = features[positive_indexes]
    negative_features = features[negative_indexes]

    similarities = torch.sum(torch.mul(positive_features, negative_features), 1)
    features_dim = features.shape[1]

    first_moment = torch.mean(similarities)
    second_moment = torch.mean(torch.pow(similarities, 2))

    loss = torch.pow(first_moment, 2) + torch.clamp(second_moment - 1./features_dim, min=0.)

    return loss


def weights_orthogonality(weights, margin=0.):
    """Regularization forcing the weights to be disimilar.

    :param weights: Learned parameters of shape (n_classes, n_features).
    :param margin: Margin to force even more the orthogonality.
    :return: A float scalar loss.
    """
    normalized_weights = F.normalize(weights, dim=1, p=2)
    similarities = torch.mm(normalized_weights, normalized_weights.t())

    # We are ignoring the diagonal made of identity similarities:
    similarities = similarities[~torch.eye(similarities.shape[0]).byte()]

    return torch.mean(F.relu(similarities + margin))


def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )


def ucir_ranking(logits, targets, n_classes, task_size, nb_negatives=2, margin=0.2):
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
    negative_indexes = old_logits[..., nb_old_classes:].argsort(
        dim=1, descending=True
    )[..., :nb_negatives] + nb_old_classes
    new_values = old_logits[torch.arange(len(old_logits)).view(-1, 1), negative_indexes].view(-1)

    return F.margin_ranking_loss(
        old_values, new_values, -torch.ones(len(old_values)).to(logits.device), margin=margin
    )


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
