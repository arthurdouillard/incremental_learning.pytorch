import glob
import json
import math
import os
import statistics

import matplotlib.pyplot as plt


def get_template_results(args):
    return {"config": args, "results": []}


def get_save_folder(model, date, label):
    year_month, day = date[:6], date[6:]
    week_number = math.ceil(int(day) / 7)

    folder_path = os.path.join(
        "results", "dev", model, year_month, "week_{}".format(week_number),
        "{}_{}".format(date, label)
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_results(results, label, model, date, run_id, seed):
    del results["config"]["device"]

    folder_path = get_save_folder(model, date, label)

    file_path = "run_{}_.json".format(seed)

    with open(os.path.join(folder_path, file_path), "w+") as f:
        try:
            json.dump(results, f, indent=2)
        except Exception:
            print("Failed to dump exps on json file.")


def extract(paths, metric="avg_inc", nb_classes=None):
    """Extract accuracy logged in the various log files.

    :param paths: A path or a list of paths to a json file.
    :param avg_inc: Boolean specifying whether to use the accuracy or the average
                    incremental accuracy as defined in iCaRL.
    :return: A list of runs. Each runs is a list of (average incremental) accuracies.
    """
    if not isinstance(paths, list):
        paths = [paths]

    score_plot, score_tab = [], []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        if metric in ("avg_inc", "accuracy"):
            score_plot.append([100 * task["accuracy"]["total"] for task in data["results"]])
        elif metric == "accuracy_top5":
            score_plot.append([100 * task["accuracy_top5"]["total"] for task in data["results"]])
        elif metric == "old_accuracy":
            score_plot.append([100 * task.get("old_accuracy", 0.) for task in data["results"]])
        elif metric == "new_accuracy":
            score_plot.append([100 * task.get("new_accuracy", 0.) for task in data["results"]])
        elif metric == "unseen":
            score_plot.append(
                [100 * task.get("unseen_classes_accuracy", 0.) for task in data["results"]]
            )
        elif metric == "seen":
            score_plot.append(
                [100 * task.get("seen_classes_accuracy", 0.) for task in data["results"]]
            )
        else:
            raise ValueError("bouh")

        if metric in ("avg_inc", "accuracy", "accuracy_top5", "old_accuracy", "new_accuracy"):
            score_tab.append(score_plot[-1])
        elif metric == "avg_cls":
            accs = []
            for class_id in range(nb_classes):
                class_accuracies = [
                    100 * task["accuracy_per_class"]["{:02d}-{:02d}".format(class_id, class_id)]
                    for task in data["results"]
                    if "{:02d}-{:02d}".format(class_id, class_id) in task["accuracy_per_class"]
                ]
                if len(class_accuracies) > 0:
                    accs.append(statistics.mean(class_accuracies))

            score_tab.append(accs)

    return score_plot, score_tab


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


def compute_unique_score(runs_accs, skip_first=False, first_n_steps=None):
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
        if first_n_steps:
            means.append(sum(run[start:first_n_steps]) / len(run[start:first_n_steps]))
        else:
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


def plot(
    results,
    increment,
    total,
    initial_increment=None,
    x_ticks=None,
    title="",
    path_to_save=None,
    max_acc=100,
    min_acc=0,
    first_n_steps=None,
    figsize=(10, 5),
    metric="avg_inc",
    zeroshot=False,
    ylabel="Accuracy over seen classes"
):
    """Plotting utilities to visualize several experiments.

    :param results: A list of dict composed of a "path", a "label", an optional
                    "average incremental", an optional "skip_first".
    :param increment: The increment of classes per task.
    :param total: The total number of classes.
    :param initial_increment: Increment initial, default to 0.
    :param title: Plot title.
    :param path_to_save: Optional path where to save the image.
    """
    plt.figure(figsize=figsize)

    initial_increment = initial_increment or increment
    x = list(range(initial_increment, total + 1, increment))

    for result in results:
        path = result.get("path", "")
        label = result.get("label", path.rstrip("/").split("/")[-1])
        skip_first = result.get("skip_first", False)
        kwargs = result.get("kwargs", {})

        if result.get("hidden", False):
            continue

        if path:
            if "*" in path:
                path = glob.glob(path)
            elif os.path.isdir(path):
                path = glob.glob(os.path.join(path, "*.json"))

            score_plot, score_tab = extract(path, metric=metric, nb_classes=total)
        else:
            score_plot = result["runs_accs"]
            score_tab = score_plot

        means, stds = aggregate(score_plot)

        if first_n_steps is not None:
            x, means, stds = x[:first_n_steps], means[:first_n_steps], stds[:first_n_steps]

        unique_score, unique_std = compute_unique_score(
            score_tab, skip_first=skip_first, first_n_steps=first_n_steps
        )

        label = "{label} ({avg})".format(
            label=label, avg=unique_score + unique_std, last=round(means[-1], 2)
        )

        try:
            bar = plt.errorbar(x, means, stds, label=label, marker="o", markersize=3, **kwargs)
        except Exception:
            print(x)
            print(means)
            print(stds)
            print(label)
            raise

        if zeroshot:
            unseen_accs, _ = extract(path, "unseen", nb_classes=total)
            plt.plot(
                x[:-1], unseen_accs[0][:-1], linestyle='dashed', color=bar.lines[0].get_color()
            )
            seen_accs, _ = extract(path, "seen", nb_classes=total)
            plt.plot(x, seen_accs[0], linestyle='dotted', color=bar.lines[0].get_color())

    plt.legend(loc="upper right")
    plt.xlabel("Number of classes")
    plt.ylabel(ylabel)
    plt.title(title)

    for y in range(min_acc, max_acc + 1, 10):
        plt.axhline(y=y, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
    plt.yticks(list(range(min_acc, max_acc + 1, 10)))

    x_ticks = x_ticks or increment
    plt.xticks(list(range(initial_increment, total + 1, x_ticks)))

    if path_to_save:
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        plt.savefig(path_to_save)
    plt.show()
