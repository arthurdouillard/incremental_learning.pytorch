import numpy as np


def get_class_weights(dataset, log=False, **kwargs):
    targets = dataset.y

    class_sample_count = np.unique(targets, return_counts=True)[1]
    weights = 1. / class_sample_count

    min_w = weights.min()
    weights = weights / min_w

    if log:
        weights = np.log(weights)

    return np.clip(weights, a_min=1., a_max=None)
