import collections
import logging

import torch
from torch import nn
from tqdm import tqdm

from inclearn.lib.network import hook

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
                gradcam_grad=grad,
                gradcam_act=act
            )
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


def perclass_loop(
    inc_dataset,
    class_ids,
    devices,
    network,
    n_epochs,
    optimizer,
    loss_function,
    task,
    n_tasks,
    target_to_word,
    scheduler=None,
):
    if len(devices) > 1:
        logger.info("Duplicating model on {} gpus.".format(len(devices)))
        training_network = nn.DataParallel(network, devices)
    else:
        training_network = network

    for epoch in range(n_epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(class_ids, ascii=True, bar_format="{desc}: {percentage:3.0f}%")

        for index, class_id in enumerate(prog_bar, start=1):
            loader = inc_dataset.get_custom_loader([class_id], mode="train", data_source="train")[1]

            class_prog_bar = tqdm(loader, ascii=True, bar_format="{bar} | {percentage:3.0f}%")

            visual_features = []
            semantic_features = []

            optimizer.zero_grad()

            for input_dict in class_prog_bar:
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                words = target_to_word(targets)

                outputs = training_network([inputs.to(devices[0]), words.to(devices[0])])

                visual_features.append(outputs["features"])
                semantic_features.append(outputs["word_embeddings"])

            visual_features = torch.cat(visual_features)
            semantic_features = torch.cat(semantic_features)

            loss = loss_function(visual_features, semantic_features)

            loss.backward()
            optimizer.step()

            metrics["gmm_loss"] += loss.item()

            _print_metrics(metrics, prog_bar, epoch, n_epochs, index, task, n_tasks)

        if scheduler:
            scheduler.step(epoch)


def _print_metrics(metrics, prog_bar, epoch, nb_epochs, nb_batches, task, n_tasks):
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
