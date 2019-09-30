import copy
import json
import os
import random
import statistics
import time

import numpy as np
import torch
import yaml

from inclearn.lib import factory, metrics, results_utils, utils


def train(args):
    autolabel = _set_up_options(args)
    if args["autolabel"]:
        args["label"] = autolabel
        print("Auto label: {}.".format(autolabel))

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()

    avg_inc_accs = []

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    for i, seed in enumerate(seed_list):
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        avg_inc_acc, last_acc, forgetting = _train(args, start_date, orders[i], i)
        avg_inc_accs.append(avg_inc_acc)
        print("Training finished in {}s.".format(int(time.time() - start_time)))

    print("Label was: {}".format(args["label"]))
    print(
        "Results done on {} seeds: avg: {}{}, last: {}, forgetting: {}".format(
            len(seed_list), round(statistics.mean(avg_inc_accs) * 100, 2), " +/- " +
            str(round(statistics.stdev(avg_inc_accs) * 100, 2)) if len(avg_inc_accs) > 1 else "",
            round(last_acc * 100, 2), round(forgetting * 100, 2)
        )
    )

    return avg_inc_accs


def _train(args, start_date, class_order, run_id):
    _set_seed(args["seed"])

    factory.set_device(args)

    inc_dataset = factory.get_data(args, class_order)
    args["classes_order"] = inc_dataset.class_order

    model = factory.get_model(args)
    model.inc_dataset = inc_dataset

    results = results_utils.get_template_results(args)

    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger()

    if args["label"] is not None:
        folder_results = os.path.join(
            "results", "dev", args["model"], "{}_{}".format(start_date, args["label"])
        )
        os.makedirs(folder_results, exist_ok=True)
        model.folder_result = folder_results
    else:
        model.folder_result = None

    for _ in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["total_n_classes"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=task_info["max_task"]
        )

        model.eval()
        model.before_task(train_loader, val_loader)
        print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        model.train_task(train_loader, val_loader)
        model.eval()
        model.after_task(inc_dataset)

        print("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypred, ytrue = model.eval_task(test_loader)
        metric_logger.log_task(ypred, ytrue)

        print(metric_logger.last_results)
        results["results"].append(metric_logger.last_results)

        memory = model.get_memory()
        memory_val = model.get_val_memory()

    print(
        "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
    )
    if args["label"] is not None:
        results_utils.save_results(results, args["label"], args["model"], start_date, run_id)

    del model
    del inc_dataset

    avg_inc_acc = results["results"][-1]["incremental_accuracy"]
    last_acc = results["results"][-1]["accuracy"]["total"]
    forgetting = results["results"][-1]["forgetting"]

    return avg_inc_acc, last_acc, forgetting


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.


def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)


def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))
