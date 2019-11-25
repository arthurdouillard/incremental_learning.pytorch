import collections

import numpy as np
import torch


class MetricLogger:

    def __init__(self):
        self.metrics = collections.defaultdict(list)

    def log_task(self, ypreds, ytrue):
        self.metrics["accuracy"].append(
            accuracy_per_task(ypreds, ytrue, task_size=10, topk=1)
        )  # FIXME various task size
        self.metrics["accuracy_top5"].append(
            accuracy_per_task(ypreds, ytrue, task_size=None, topk=5)
        )
        self.metrics["accuracy_per_class"].append(
            accuracy_per_task(ypreds, ytrue, task_size=1, topk=1)
        )
        self.metrics["incremental_accuracy"].append(incremental_accuracy(self.metrics["accuracy"]))
        self.metrics["incremental_accuracy_top5"].append(
            incremental_accuracy(self.metrics["accuracy_top5"])
        )
        self.metrics["forgetting"].append(forgetting(self.metrics["accuracy"]))

    @property
    def last_results(self):
        return {
            "accuracy": self.metrics["accuracy"][-1],
            "incremental_accuracy": self.metrics["incremental_accuracy"][-1],
            "accuracy_top5": self.metrics["accuracy_top5"][-1],
            "incremental_accuracy_top5": self.metrics["incremental_accuracy_top5"][-1],
            "forgetting": self.metrics["forgetting"][-1],
            "accuracy_per_class": self.metrics["accuracy_per_class"][-1]
        }


def accuracy_per_task(ypreds, ytrue, task_size=10, topk=1):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    all_acc = {}

    all_acc["total"] = accuracy(ypreds, ytrue, topk=topk)

    if task_size is not None:
        for class_id in range(0, np.max(ytrue), task_size):
            idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

            label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
            )
            all_acc[label] = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)

    return all_acc


def accuracy(output, targets, topk=1):
    """Computes the precision@k for the specified values of k"""
    output, targets = torch.tensor(output), torch.tensor(targets)

    batch_size = targets.shape[0]
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].view(-1).float().sum(0).item()
    return round(correct_k / batch_size, 3)


def incremental_accuracy(accuracies):
    """Computes the average incremental accuracy as described in iCaRL.

    It is the average of the current task accuracy (tested on 0-X) with the
    previous task accuracy.

    :param acc_dict: A list TODO
    """
    return sum(task_acc["total"] for task_acc in accuracies) / len(accuracies)


def forgetting(accuracies):
    if len(accuracies) == 1:
        return 0.

    last_accuracies = accuracies[-1]
    usable_tasks = last_accuracies.keys()

    forgetting = 0.
    for task in usable_tasks:
        if task == "total":
            continue

        max_task = 0.

        for task_accuracies in accuracies[:-1]:
            if task in task_accuracies:
                max_task = max(max_task, task_accuracies[task])

        forgetting += max_task - last_accuracies[task]

    return forgetting / len(usable_tasks)
