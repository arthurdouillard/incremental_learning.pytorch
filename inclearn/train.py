import copy
import os
import random
import time

import numpy as np
import torch
import yaml

from inclearn.lib import factory, results_utils, utils


def train(args):
    _set_up_options(args)

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()
        _train(args, start_date)
        print("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(args, start_date):
    _set_seed(args["seed"])

    factory.set_device(args)

    inc_dataset = factory.get_data(args)
    args["classes_order"] = inc_dataset.class_order

    model = factory.get_model(args)
    model.inc_dataset = inc_dataset

    results = results_utils.get_template_results(args)

    memory, memory_val = None, None

    for _ in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
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
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
        print(acc_stats)
        results["results"].append(acc_stats)

        memory = model.get_memory()
        memory_val = model.get_val_memory()

    print(
        "Average Incremental Accuracy: {}.".format(
            results_utils.compute_avg_inc_acc(results["results"])
        )
    )

    if args["name"] is not None:
        results_utils.save_results(results, args["name"], args["model"], start_date)

    del model
    del inc_dataset
    torch.cuda.empty_cache()


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.


def _set_up_options(args):
    options_paths = args["options"] or []

    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        with open(option_path) as f:
            options = yaml.load(f, Loader=yaml.FullLoader)

        args.update(options)
