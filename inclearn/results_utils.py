import json
import os

import matplotlib.pyplot as plt

from inclearn import utils


def get_template_results(args):
    return {
        "config": args,
        "results": []
    }


def save_results(results, label):
    file_path = "{}_{}.json".format(utils.get_date(), label)
    with open(os.path.join("results", file_path), "w+") as f:
        json.dump(results, f)



def plot(paths, save_path=None):
    plt.figure(figsize=(10, 5))

    for path in paths:
        label = path.split("/")[-1].split(".")[0]
        x, y = _extract(path)

        plt.plot(x, y, label=label)
    plt.legend(loc="upper right")
    plt.xlabel("Number of classes")
    plt.ylabel("Accuracy")

    for i in range(10, 101, 10):
        plt.axhline(y=i, color='black', linestyle='dashed', linewidth=1, alpha=0.2)
    plt.yticks([i for i in range(10, 101, 10)])
    plt.xticks([i for i in range(10, len(x) * 10 + 1, 10)])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# -----

def _sorting(x):
    if x != "total":
        return x.split("-")[1]
    return x

def _get_number_of_classes(task):
    max_task = sorted(task.keys(), key=lambda x: _sorting(x), reverse=True)[1]
    return int(max_task.split("-")[1]) + 1

def _extract(path):
    with open(path) as f:
        stats = json.load(f)

    x = [_get_number_of_classes(task) for task in stats]
    y = [100 * task["total"] for task in stats]

    return x, y
