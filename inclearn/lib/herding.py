import torch
from torch.nn import functional as F


def closest_to_mean(features):
    F.normalize(features)
    class_mean = torch.mean(features, dim=0, keepdim=False)

    return



def l2_distance(x, y):
    return (x - y).norm()
