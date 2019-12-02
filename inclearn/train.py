import copy
import json
import logging
import os
import random
import statistics
import time

import numpy as np
import torch
import yaml

from inclearn.lib import factory
from inclearn.lib import logger as logger_lib
from inclearn.lib import metrics, results_utils, utils

logger = logging.getLogger(__name__)


def train(args):
    logger_lib.set_logging_level(args["logging"])

    autolabel = _set_up_options(args)
    if args["autolabel"]:
        args["label"] = autolabel

    if args["label"]:
        logger.info("Label: {}".format(args["label"]))

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

        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i):
            yield avg_inc_acc, last_acc, forgetting

        avg_inc_accs.append(avg_inc_acc)
        logger.info("Training finished in {}s.".format(int(time.time() - start_time)))

    logger.info("Label was: {}".format(args["label"]))
    logger.info(
        "Results done on {} seeds: avg: {}{}, last: {}, forgetting: {}".format(
            len(seed_list), round(statistics.mean(avg_inc_accs) * 100, 2), " +/- " +
            str(round(statistics.stdev(avg_inc_accs) * 100, 2)) if len(avg_inc_accs) > 1 else "",
            round(last_acc * 100, 2), round(forgetting * 100, 2)
        )
    )
    logger.info("Individual results: {}".format([round(100 * acc, 2) for acc in avg_inc_accs]))


def _train(args, start_date, class_order, run_id):
    _set_seed(args["seed"], args["threads"], args["no_benchmark"])

    factory.set_device(args)

    inc_dataset = factory.get_data(args, class_order)
    args["classes_order"] = inc_dataset.class_order

    model = factory.get_model(args)
    model.inc_dataset = inc_dataset

    if args["label"]:
        results_folder = results_utils.get_save_folder(args["model"], start_date, args["label"])
        if args["save_model"]:
            logger.info("Model will be save at this rythm: {}.".format(args["save_model"]))

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

    for task_id in range(inc_dataset.n_tasks):
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
        model.before_task(train_loader, val_loader if val_loader else test_loader)

        if task_id == 0 and args["resume"] is not None:
            network_path = args["resume"]
            model.network = network_path
            logger.info("Skipping training phase because reloading pretrained model.")
        else:
            logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            model.train()
            model.train_task(train_loader, val_loader if val_loader else test_loader)
        model.eval()

        if args["label"] and (
            args["save_model"] == "task" or
            (args["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (args["save_model"] == "first" and task_id == 0)
        ):
            save_name = "net_{}_task_{}.pth".format(run_id, task_id)
            logger.info("Saving model: {}.".format(save_name))
            torch.save(model.network, os.path.join(results_folder, save_name))

        model.after_task(inc_dataset)

        if args["label"] and (
            args["save_model"] == "task" or
            (args["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (args["save_model"] == "first" and task_id == 0)
        ):
            meta_save_name = "meta_{}_task_{}.pth".format(run_id, task_id)
            logger.info("Saving meta data: {}.".format(meta_save_name))
            model.save_metadata(os.path.join(results_folder, meta_save_name))

        logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypreds, ytrue = model.eval_task(test_loader)
        metric_logger.log_task(ypreds, ytrue)

        if args["label"]:
            logger.info(args["label"])
        logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
        logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
        logger.info(
            "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
        )
        logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
        logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
        results["results"].append(metric_logger.last_results)

        avg_inc_acc = results["results"][-1]["incremental_accuracy"]
        last_acc = results["results"][-1]["accuracy"]["total"]
        forgetting = results["results"][-1]["forgetting"]
        yield avg_inc_acc, last_acc, forgetting

        memory = model.get_memory()
        memory_val = model.get_val_memory()

    logger.info(
        "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
    )
    if args["label"] is not None:
        results_utils.save_results(results, args["label"], args["model"], start_date, run_id)

    del model
    del inc_dataset


def _set_seed(seed, nb_threads, no_benchmark):
    logger.info("Set seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if no_benchmark:
        logger.warning("CUDA algos are not determinists but faster!")
    else:
        logger.warning("CUDA algos are determinists but very slow!")
    torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
    torch.set_num_threads(nb_threads)


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
