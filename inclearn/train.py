import copy
import json
import logging
import os
import pickle
import random
import statistics
import sys
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
        try:
            os.system("echo '\ek{}\e\\'".format(args["label"]))
        except:
            pass
    if args["resume"] and not os.path.exists(args["resume"]):
        raise IOError(f"Saved model {args['resume']} doesn't exist.")

    if args["save_model"] != "never" and args["label"] is None:
        raise ValueError(f"Saving model every {args['save_model']} but no label was specified.")

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    avg_inc_accs, last_accs, forgettings = [], [], []
    for i, seed in enumerate(seed_list):
        logger.warning("Launching run {}/{}".format(i + 1, len(seed_list)))
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i):
            yield avg_inc_acc, last_acc, forgetting, False

        avg_inc_accs.append(avg_inc_acc)
        last_accs.append(last_acc)
        forgettings.append(forgetting)

        logger.info("Training finished in {}s.".format(int(time.time() - start_time)))
        yield avg_inc_acc, last_acc, forgetting, True

    logger.info("Label was: {}".format(args["label"]))

    logger.info(
        "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
            len(seed_list), _aggregate_results(avg_inc_accs), _aggregate_results(last_accs),
            _aggregate_results(forgettings)
        )
    )
    logger.info("Individual results avg: {}".format([round(100 * acc, 2) for acc in avg_inc_accs]))
    logger.info("Individual results last: {}".format([round(100 * acc, 2) for acc in last_accs]))
    logger.info(
        "Individual results forget: {}".format([round(100 * acc, 2) for acc in forgettings])
    )

    logger.info(f"Command was {' '.join(sys.argv)}")


def _train(args, start_date, class_order, run_id):
    _set_global_parameters(args)
    inc_dataset, model = _set_data_model(args, class_order)
    results, results_folder = _set_results(args, start_date)

    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )

    for task_id in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(task_info)

        # ---------------
        # 1. Prepare Task
        # ---------------
        model.eval()
        model.before_task(train_loader, val_loader if val_loader else test_loader)

        # -------------
        # 2. Train Task
        # -------------
        _train_task(args, model, train_loader, val_loader, test_loader, run_id, task_id, task_info)

        # ----------------
        # 3. Conclude Task
        # ----------------
        model.eval()
        _after_task(args, model, inc_dataset, run_id, task_id, results_folder)

        # ------------
        # 4. Eval Task
        # ------------
        logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypreds, ytrue = model.eval_task(test_loader)
        metric_logger.log_task(
            ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
        )

        if args["dump_predictions"] and args["label"]:
            os.makedirs(
                os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
            )
            with open(
                os.path.join(
                    results_folder, "predictions_{}".format(run_id),
                    str(task_id).rjust(len(str(30)), "0") + ".pkl"
                ), "wb+"
            ) as f:
                pickle.dump((ypreds, ytrue), f)

        if args["label"]:
            logger.info(args["label"])
        logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
        logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
        logger.info(
            "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
        )
        logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
        logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
        logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
        if task_id > 0:
            logger.info(
                "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["old_accuracy"],
                    metric_logger.last_results["avg_old_accuracy"]
                )
            )
            logger.info(
                "New accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["new_accuracy"],
                    metric_logger.last_results["avg_new_accuracy"]
                )
            )
        if args.get("all_test_classes"):
            logger.info(
                "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
            )
            logger.info(
                "unSeen classes: {:.2f}.".format(
                    metric_logger.last_results["unseen_classes_accuracy"]
                )
            )

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
        results_utils.save_results(
            results, args["label"], args["model"], start_date, run_id, args["seed"]
        )

    del model
    del inc_dataset


# ------------------------
# Lifelong Learning phases
# ------------------------


def _train_task(config, model, train_loader, val_loader, test_loader, run_id, task_id, task_info):
    if config["resume"] is not None and os.path.isdir(config["resume"]) \
       and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
        model.load_parameters(config["resume"], run_id)
        logger.info(
            "Skipping training phase {} because reloading pretrained model.".format(task_id)
        )
    elif config["resume"] is not None and os.path.isfile(config["resume"]) and \
            os.path.exists(config["resume"]) and task_id == 0:
        # In case we resume from a single model file, it's assumed to be from the first task.
        model.network = config["resume"]
        logger.info(
            "Skipping initial training phase {} because reloading pretrained model.".
            format(task_id)
        )
    else:
        logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        model.train_task(train_loader, val_loader if val_loader else test_loader)


def _after_task(config, model, inc_dataset, run_id, task_id, results_folder):
    if config["resume"] and os.path.isdir(config["resume"]) and not config["recompute_meta"] \
       and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
        model.load_metadata(config["resume"], run_id)
    else:
        model.after_task_intensive(inc_dataset)

    model.after_task(inc_dataset)

    if config["label"] and (
        config["save_model"] == "task" or
        (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
        (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        model.save_metadata(results_folder, run_id)


# ----------
# Parameters
# ----------


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(config["model"], start_date, config["label"])
    else:
        results_folder = None

    if config["save_model"]:
        logger.info("Model will be save at this rythm: {}.".format(config["save_model"]))

    results = results_utils.get_template_results(config)

    return results, results_folder


def _set_data_model(config, class_order):
    inc_dataset = factory.get_data(config, class_order)
    config["classes_order"] = inc_dataset.class_order

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model


def _set_global_parameters(config):
    _set_seed(config["seed"], config["threads"], config["no_benchmark"], config["detect_anomaly"])
    factory.set_device(config)


def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
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
    if detect_anomaly:
        logger.info("Will detect autograd anomaly.")
        torch.autograd.set_detect_anomaly(detect_anomaly)


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


# ----
# Misc
# ----


def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    if len(list_results) > 1:
        res = res + " +/- " + str(round(statistics.stdev(list_results) * 100, 2))
    return res
