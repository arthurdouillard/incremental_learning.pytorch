import copy
import random

import numpy as np
import torch

from inclearn import factory, results_utils, utils


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    _set_seed(args["seed"])

    factory.set_device(args)

    train_set = factory.get_data(args, train=True)
    test_set = factory.get_data(args, train=False, classes_order=train_set.classes_order)

    train_loader, val_loader = train_set.get_loader(args["validation"])
    test_loader, _ = test_set.get_loader()
    #val_loader = test_loader

    model = factory.get_model(args)

    results = results_utils.get_template_results(args)

    for task in range(0, train_set.total_n_classes // args["increment"]):
        if args["max_task"] == task:
            break

        # Setting current task's classes:
        train_set.set_classes_range(low=task * args["increment"],
                                    high=(task + 1) * args["increment"])
        test_set.set_classes_range(high=(task + 1) * args["increment"])

        model.set_task_info(
            task,
            train_set.total_n_classes,
            args["increment"],
            len(train_set),
            len(test_set)
        )

        model.before_task(train_loader, val_loader)
        print("train", task * args["increment"], (task + 1) * args["increment"])
        model.train_task(train_loader, val_loader)
        model.after_task(train_loader)

        ypred, ytrue = model.eval_task(test_loader)
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
        print(acc_stats)
        results["results"].append(acc_stats)

        memory_indexes = model.get_memory_indexes()
        train_set.set_memory(memory_indexes)

    if args["name"]:
        results_utils.save_results(results, args["name"])

    del model
    del train_set
    del test_set
    torch.cuda.empty_cache()


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
