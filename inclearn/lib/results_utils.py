import glob
import json
import math
import os

import matplotlib.pyplot as plt

from inclearn.lib import utils


def get_template_results(args):
    return {"config": args, "results": []}


def save_results(results, label):
    del results["config"]["device"]

    folder_path = os.path.join("results", "{}_{}".format(utils.get_date(), label))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = "{}_{}_.json".format(utils.get_date(), results["config"]["seed"])
    with open(os.path.join(folder_path, file_path), "w+") as f:
        json.dump(results, f, indent=2)


def extract(paths, avg_inc=False):
    """Extract accuracy logged in the various log files.

    :param paths: A path or a list of paths to a json file.
    :param avg_inc: Boolean specifying whether to use the accuracy or the average
                    incremental accuracy as defined in iCaRL.
    :return: A list of runs. Each runs is a list of (average incremental) accuracies.
    """
    if not isinstance(paths, list):
        paths = [paths]

    runs_accs = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data["results"][0], dict):
            accs = [100 * task["total"] for task in data["results"]]
        elif isinstance(data["results"][0], float):
            accs = [100 * task_acc for task_acc in data["results"]]
        else:
            raise NotImplementedError(type(data["results"][0]))

        if avg_inc:
            raise NotImplementedError("Deprecated")

        runs_accs.append(accs)

    return runs_accs


def compute_avg_inc_acc(results):
    """Computes the average incremental accuracy as defined in iCaRL.

    The average incremental accuracies at task X are the average of accuracies
    at task 0, 1, ..., and X.

    :param accs: A list of dict for per-class accuracy at each step.
    :return: A float.
    """
    tasks_accuracy = [r["total"] for r in results]
    return sum(tasks_accuracy) / len(tasks_accuracy)


def aggregate(runs_accs):
    """Aggregate results of several runs into means & standard deviations.

    :param runs_accs: A list of runs. Each runs is a list of (average
                      incremental) accuracies.
    :return: A list of means, and a list of standard deviations.
    """
    means = []
    stds = []

    n_runs = len(runs_accs)
    for i in range(len(runs_accs[0])):
        ith_value = [runs_accs[j][i] for j in range(n_runs)]

        mean = sum(ith_value) / n_runs
        std = math.sqrt(sum(math.pow(mean - i, 2) for i in ith_value) / n_runs)

        means.append(mean)
        stds.append(std)

    return means, stds


def compute_unique_score(runs_accs, skip_first=False):
    """Computes the average of the (average incremental) accuracies to get a
    unique score.

    :param runs_accs: A list of runs. Each runs is a list of (average
                      incremental) accuracies.
    :param skip_first: Whether to skip the first task accuracy as advised in
                       End-to-End Incremental Accuracy.
    :return: A unique score being the average of the (average incremental)
             accuracies, and a standard deviation.
    """
    start = int(skip_first)

    means = []
    for run in runs_accs:
        means.append(sum(run[start:]) / len(run[start:]))

    mean_of_mean = sum(means) / len(means)
    if len(runs_accs) == 1:  # One run, probably a paper, don't compute std:
        std = ""
    else:
        std = math.sqrt(sum(math.pow(mean_of_mean - i, 2) for i in means) / len(means))
        std = " ± " + str(round(std, 2))

    return str(round(mean_of_mean, 2)), std


def get_max_label_length(results):
    return max(len(r.get("label", r["path"])) for r in results)


def plot(results, increment, total, title="", path_to_save=None):
    """Plotting utilities to visualize several experiments.

    :param results: A list of dict composed of a "path", a "label", an optional
                    "average incremental", an optional "skip_first".
    :param increment: The increment of classes per task.
    :param total: The total number of classes.
    :param title: Plot title.
    :param path_to_save: Optional path where to save the image.
    """
    plt.figure(figsize=(10, 5))

    x = list(range(increment, total + 1, increment))

    max_label_length = get_max_label_length(results) + 4

    for result in results:
        path = result["path"]
        label = result.get("label", path)
        from_paper = "[paper] " if result.get("from_paper", False) else "[me]     "
        avg_inc = result.get("average_incremental", False)
        skip_first = result.get("skip_first", False)
        kwargs = result.get("kwargs", {})

        if result.get("hidden", False):
            continue

        if "*" in path:
            path = glob.glob(path)
        elif os.path.isdir(path):
            path = glob.glob(os.path.join(path, "*.json"))

        runs_accs = extract(path, avg_inc=avg_inc)
        means, stds = aggregate(runs_accs)

        unique_score, unique_std = compute_unique_score(runs_accs, skip_first=skip_first)

        label = "{mode}{label}(avg: {avg}, last: {last})".format(
            mode=from_paper,
            label=label.ljust(max_label_length, " "),
            avg=unique_score + unique_std,
            last=round(means[-1], 2)
        )

        try:
            plt.errorbar(x, means, stds, label=label, marker="o", markersize=3, **kwargs)
        except Exception:
            print(x)
            print(means)
            print(stds)
            print(label)
            raise

    plt.legend(loc="upper right")
    plt.xlabel("Number of classes")
    plt.ylabel("Accuracy over seen classes")
    plt.title(title)

    for i in range(10, total + 1, 10):
        plt.axhline(y=i, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
    plt.yticks([i for i in range(10, total + 1, 10)])
    plt.xticks([i for i in range(10, len(x) * increment + 1, 10)])

    if path_to_save:
        plt.savefig(path_to_save)
    plt.show()
