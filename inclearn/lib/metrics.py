import numpy as np


class MetricLogger:
    def __init__(self):
        self._accuracy_per_task = []
        self._accuracy = []
        self._incremental_accuracy = []

    def add_task(ypred, ytrue, task_name):
        pass



def compute_metrics(ypred, ytrue, task_size=10):
    metrics = {}

    metrics["accuracy"] = accuracy(ypred, ytrue, task_size=task_size)
    metrics["incremental_accuracy"] = incremental_accuracy(metrics["accuracy"])


def accuracy(ypred, ytrue, task_size=10):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    all_acc = {}

    all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(
                np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
        )[0]

        label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc



def incremental_accuracy(acc_dict):
    """Computes the average incremental accuracy as described in iCaRL.

    It is the average of the current task accuracy (tested on 0-X) with the
    previous task accuracy.

    :param acc_dict: A dict TODO
    """
    v, c = 0., 0

    pass
