import copy
import random
import time

import numpy as np
import torch

from inclearn.lib import factory, results_utils, utils


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()
        _train(args)
        print("Training finished in {}s.".format(time.time() - start_time))


def _train(args):
    _set_seed(args["seed"])

    factory.set_device(args)

    inc_dataset = factory.get_data(args)
    args["classes_order"] = inc_dataset.class_order

    model = factory.get_model(args)

    results = results_utils.get_template_results(args)

    memory = None

    for _ in range(inc_dataset.n_tasks):
        task_info, train_loader, test_loader = inc_dataset.new_task(memory)
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
        model.before_task(train_loader, None)
        print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        model.train_task(train_loader, None)
        model.eval()
        model.after_task(inc_dataset)

        print("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypred, ytrue = model.eval_task(test_loader)
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
        print(acc_stats)
        results["results"].append(acc_stats)

        memory = model.get_memory()

    if args["name"]:
        results_utils.save_results(results, args["name"])

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
