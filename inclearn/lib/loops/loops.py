import collections
import logging

import torch
from torch import nn

from inclearn.lib.network import hook
from tqdm import tqdm

logger = logging.getLogger(__name__)


def single_loop(
    train_loader,
    val_loader,
    devices,
    network,
    n_epochs,
    optimizer,
    train_function,
    eval_function,
    task,
    n_tasks,
    scheduler=None,
    disable_progressbar=False,
    eval_every_x_epochs=None,
    config=None,
    early_stopping=None
):
    best_epoch, best_acc = -1, -1.
    wait = 0

    grad, act = None, None
    if len(devices) > 1:
        logger.info("Duplicating model on {} gpus.".format(len(devices)))
        training_network = nn.DataParallel(network, devices)
        if network.gradcam_hook:
            logger.info("Adding hook on multi-gpu model.")
            grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
            training_network.module.convnet.last_conv.register_backward_hook(back_hook)
            training_network.module.convnet.last_conv.register_forward_hook(for_hook)
    else:
        training_network = network

    for epoch in range(n_epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(
            train_loader,
            disable=disable_progressbar,
            ascii=True,
            bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
        )
        for batch_index, input_dict in enumerate(prog_bar, start=1):
            inputs, targets = input_dict["inputs"], input_dict["targets"]
            memory_flags = input_dict["memory_flags"]

            if grad is not None:
                _clean_list(grad)
                _clean_list(act)

            optimizer.zero_grad()
            loss = train_function(
                training_network,
                inputs,
                targets,
                memory_flags,
                metrics,
                epoch=epoch,
                epochs=n_epochs,
                gradcam_grad=grad,
                gradcam_act=act,
                config=config
            )
            if isinstance(loss, float):
                # Not a tensor, because the loss was never computed, probably
                # because of some sample requirements
                continue
            loss.backward()
            optimizer.step()

            _print_metrics(metrics, prog_bar, epoch, n_epochs, batch_index, task, n_tasks)

        if scheduler:
            scheduler.step(epoch)

        if eval_every_x_epochs and epoch != 0 and epoch % eval_every_x_epochs == 0:
            training_network.eval()
            accuracy = eval_function(training_network, val_loader)
            training_network.train()

            logger.info("Val accuracy: {}".format(accuracy))

            if accuracy > best_acc:
                best_epoch = epoch
                best_acc = accuracy
                wait = 0
            else:
                wait += 1

            if early_stopping and early_stopping["patience"] > wait:
                logger.warning("Early stopping!")
                break

    if eval_every_x_epochs:
        logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))


def _print_metrics(metrics, prog_bar, epoch, nb_epochs, nb_batches, task, n_tasks):
    failed_metrics = []
    for metric_name, metric_value in metrics.items():
        if metric_value == float('nan'):
            failed_metrics.append(metric_name)
    if len(failed_metrics) > 0:
        raise Exception(f"Following metrics were NaN: {str(failed_metrics)}!")

    pretty_metrics = ", ".join(
        "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
        for metric_name, metric_value in metrics.items()
    )

    prog_bar.set_description(
        "T{}/{}, E{}/{} => {}".format(task + 1, n_tasks, epoch + 1, nb_epochs, pretty_metrics)
    )


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
