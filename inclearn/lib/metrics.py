import collections

import numpy as np
import torch


class MetricLogger:

    def __init__(self, nb_tasks, nb_classes, increments):
        self.metrics = collections.defaultdict(list)

        self.nb_tasks = nb_tasks
        self.nb_classes = nb_classes
        self.increments = increments

        self._accuracy_matrix = np.ones((nb_classes, nb_tasks), dtype="float16") * -1
        self._task_counter = 0

    def log_task(self, ypreds, ytrue, task_size, zeroshot=False):
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

        self._update_accuracy_matrix(self.metrics["accuracy_per_class"][-1])
        self.metrics["cord"].append(cord_metric(self._accuracy_matrix))
        #self.metrics["cord_old"].append(cord_metric(self._accuracy_matrix, only="old"))
        #self.metrics["cord_new"].append(cord_metric(self._accuracy_matrix, only="new"))

        if zeroshot:
            seen_classes_indexes = np.where(ytrue < sum(self.increments[:self._task_counter + 1])
                                           )[0]
            self.metrics["seen_classes_accuracy"].append(
                accuracy(ypreds[seen_classes_indexes], ytrue[seen_classes_indexes])
            )
            unseen_classes_indexes = np.where(
                ytrue >= sum(self.increments[:self._task_counter + 1])
            )[0]
            self.metrics["unseen_classes_accuracy"].append(
                accuracy(ypreds[unseen_classes_indexes], ytrue[unseen_classes_indexes])
            )

        if self._task_counter > 0:
            self.metrics["old_accuracy"].append(old_accuracy(ypreds, ytrue, task_size))
            self.metrics["new_accuracy"].append(new_accuracy(ypreds, ytrue, task_size))

        self._task_counter += 1

    @property
    def last_results(self):
        results = {
            "task_id": len(self.metrics["accuracy"]) - 1,
            "accuracy": self.metrics["accuracy"][-1],
            "incremental_accuracy": self.metrics["incremental_accuracy"][-1],
            "accuracy_top5": self.metrics["accuracy_top5"][-1],
            "incremental_accuracy_top5": self.metrics["incremental_accuracy_top5"][-1],
            "forgetting": self.metrics["forgetting"][-1],
            "accuracy_per_class": self.metrics["accuracy_per_class"][-1],
            "cord": self.metrics["cord"][-1]
        }

        if "old_accuracy" in self.metrics:
            results.update(
                {
                    "old_accuracy": self.metrics["old_accuracy"][-1],
                    "new_accuracy": self.metrics["new_accuracy"][-1],
                    "avg_old_accuracy": np.mean(self.metrics["old_accuracy"]),
                    "avg_new_accuracy": np.mean(self.metrics["new_accuracy"]),
                }
            )
        if "seen_classes_accuracy" in self.metrics:
            results.update(
                {
                    "seen_classes_accuracy": self.metrics["seen_classes_accuracy"][-1],
                    "unseen_classes_accuracy": self.metrics["unseen_classes_accuracy"][-1],
                }
            )

        return results

    def _update_accuracy_matrix(self, new_accuracy_per_class):
        for k, v in new_accuracy_per_class.items():
            if k == "total":
                continue
            class_id = int(k.split("-")[0])
            self._accuracy_matrix[class_id, self._task_counter] = v


def cord_metric(accuracy_matrix, only=None):
    accuracies = []

    for class_id in range(accuracy_matrix.shape[0]):
        filled_indexes = np.where(accuracy_matrix[class_id] > -1.)[0]

        if only == "old":
            filled_indexes[1:]
        elif only == "new":
            filled_indexes[:1]

        if len(filled_indexes) == 0:
            continue
        accuracies.append(np.mean(accuracy_matrix[class_id, filled_indexes]))
    return np.mean(accuracies).item()


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
        for class_id in range(0, np.max(ytrue) + task_size, task_size):
            if class_id > np.max(ytrue):
                break

            idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

            label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
            )
            all_acc[label] = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)

    return all_acc


def old_accuracy(ypreds, ytrue, task_size):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    nb_classes = ypreds.shape[1]
    old_class_indexes = np.where(ytrue < nb_classes - task_size)[0]
    return accuracy(ypreds[old_class_indexes], ytrue[old_class_indexes], topk=1)


def new_accuracy(ypreds, ytrue, task_size):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    nb_classes = ypreds.shape[1]
    new_class_indexes = np.where(ytrue >= nb_classes - task_size)[0]
    return accuracy(ypreds[new_class_indexes], ytrue[new_class_indexes], topk=1)


def accuracy(output, targets, topk=1):
    """Computes the precision@k for the specified values of k"""
    output, targets = torch.tensor(output), torch.tensor(targets)

    batch_size = targets.shape[0]
    if batch_size == 0:
        return 0.
    nb_classes = len(np.unique(targets))
    topk = min(topk, nb_classes)

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


def forward_transfer(accuracies):
    """Measures the influence that learning a task has on the performance of future tasks.

    References:
        * Don't forget, there is more than forgetting: new metrics for Continual Learning
          Diaz et al.
          NeurIPS Workshop 2018
    """
    nb_tasks = len(accuracies)

    fwt = 0
    for i in range(nb_tasks):
        pass
