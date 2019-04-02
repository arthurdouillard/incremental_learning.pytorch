import torch

def to_onehot(targets, n_classes):
    return torch.eye(n_classes)[targets]


def _check_loss(loss):
    return not torch.isnan(loss) and loss >= 0.
