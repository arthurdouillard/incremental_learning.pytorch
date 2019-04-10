import datetime

import numpy as np
import torch


def to_onehot(targets, n_classes):
    return torch.eye(n_classes)[targets]


def _check_loss(loss):
    return not torch.isnan(loss) and loss >= 0.


def compute_accuracy(ypred, ytrue, task_size=10):
    all_acc = {}

    all_acc["total"] = (ypred == ytrue).sum() / len(ytrue)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(
                np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
        )[0]

        label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = (ypred[idxes] == ytrue[idxes]).sum() / len(idxes)

    return all_acc


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")
