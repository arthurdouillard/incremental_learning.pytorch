import json
import os

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
