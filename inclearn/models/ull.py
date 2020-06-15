import copy
import functools
import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from inclearn.lib import data, distance, factory, loops, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.lib.network.word import Word2vec
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class ULL(ICarl):

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self.unsupervised_training = args["unsupervised_training"]
        self.new_supervised_training = args["new_supervised_training"]
        self.all_supervised_training = args["all_supervised_training"]
        self._weight_decay = args["weight_decay"]

        self._finetuning_config = args.get("finetuning", {})

        # Losses definition
        self.memory_bank = args["memory_bank"]
        self.nce_loss = args["nce_loss"]

        self.rotations_prediction = args.get("rotation_prediction")

        logger.info("Initializing ULL")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {"type": "fc"}),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=True,
            return_features=True,
            rotations_predictor=bool(self.rotations_prediction)
        )

        self._n_classes = 0
        self._old_model = None

        self._data_memory, self._targets_memory = None, None
        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._fixed_memory = args.get("fixed_memory", True)
        self._memory_size = args["memory_size"]
        self._herding_selection = {"type": "icarl"}

    def _before_task(self, train_loader, val_loader):
        utils.add_new_weights(
            self._network, {"type": "basic"}, self._n_classes, self._task_size, self.inc_dataset
        )
        self._n_classes += self._task_size

        self._optimizer = factory.get_optimizer(
            [{
                "params": self._network.convnet.parameters(),
            }], self.unsupervised_training["optimizer"], self.unsupervised_training["lr"],
            self._weight_decay
        )

        self._scheduler = factory.get_lr_scheduler(
            self.unsupervised_training["scheduling"],
            self._optimizer,
            nb_epochs=self.unsupervised_training["epochs"],
            lr_decay=self.unsupervised_training.get("lr_decay", 0.1),
            task=self._task
        )

    def _train_task(self, train_loader, val_loader):
        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self.unsupervised_training["epochs"],
            self._optimizer,
            scheduler=self._scheduler,
            train_function=self._unsupervised_forward,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks,
            disable_progressbar=self._disable_progressbar
        )

        if self.new_supervised_training:
            logger.info("Finetuning new")
            self.finetuning(
                train_loader, val_loader, self.new_supervised_training,
                [{
                    "params": self._network.classifier.new_weights
                }]
            )
        if self.all_supervised_training:
            logger.info("Finetuning all")
            self.finetuning(
                train_loader, val_loader, self.all_supervised_training,
                [{
                    "params": self._network.classifier.parameters()
                }]
            )

    def finetuning(self, train_loader, val_loader, config, params):
        if config["sampling"] == "undersampling" or \
           (config["sampling"] == "next_undersampling" and self._task > 0):
            self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                self.inc_dataset, self._herding_indexes
            )
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())
        elif config["sampling"] == "new":
            class_ids = list(range(self._n_classes - self._task_size, self._n_classes))
            _, loader = self.inc_dataset.get_custom_loader([class_ids], mode="train")
        else:
            loader = train_loader

        optimizer = factory.get_optimizer(
            params, config["optimizer"], config["lr"], self._weight_decay
        )
        scheduler = factory.get_lr_scheduler(
            config["scheduling"],
            optimizer,
            nb_epochs=config["epochs"],
            lr_decay=config.get("lr_decay", 0.1),
            task=self._task
        )

        loops.single_loop(
            loader,
            val_loader,
            self._multiple_devices,
            self._network,
            config["epochs"],
            optimizer,
            scheduler=scheduler,
            train_function=self._supervised_forward,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks,
            disable_progressbar=self._disable_progressbar
        )

    def _after_task(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )
        self._old_model = self._network.copy().eval().to(self._device)
        self._network.on_task_end()

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for input_dict in loader:
            with torch.no_grad():
                logits = self._network(input_dict["inputs"].to(self._device))["logits"]

            ytrue.append(input_dict["targets"].numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _accuracy(self, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(dim=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _supervised_forward(
        self, training_network, inputs, targets, memory_flags, metrics, **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        outputs = training_network(inputs)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, targets)
        metrics["loss"] += loss.item()
        return loss

    def _unsupervised_forward(
        self, training_network, inputs, targets, memory_flags, metrics, **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        # TODO memory bank
        #outputs = training_network(inputs)
        if self.rotations_prediction:
            rot = losses.unsupervised_rotations(
                inputs, memory_flags, training_network, **self.rotations_prediction
            )
            loss = rot
            metrics["rot"] += rot.item()

        return loss


def loss(v, vv, bank, indexes, lbd=0.5, temp=1.0):
    m = bank.get(indexes)
    negs = bank.get_neg(indexes)

    losses_a = nce(m, vv, negs, temp=temp)
    losses_b = nce(m, v, negs, temp=temp)
    return torch.mean(lbd * losses_a + (1 - lbd) * losses_b)


def nce(v, vv, bank, temp=1):
    part1 = torch.log(h1(v, vv, bank, temp=temp))
    part2 = torch.log(1 - h2(vv, bank, temp=temp))

    return -(part1 + part2)


def h1(v, vv, bank, temp=1):
    """Part 1 of the L_nce equation (4)."""
    num = torch.exp(F.cosine_similarity(v, vv) / temp)

    deno_neg = torch.sum(torch.exp(torch.mm(vv, bank.t())), dim=1)
    deno = deno_neg + num

    return num / (deno + 1e-8)


def h2(v, bank, temp=1):
    """Part 2 of the L_nce equation (4)."""
    num = torch.exp(torch.bmm(v[None], bank[None].permute(0, 2, 1)).squeeze(0).sum(-1) / temp)

    deno_neg = torch.sum(torch.exp(torch.ones(len(bank)) / temp), dim=-1)
    deno = deno_neg + num

    return num / (deno + 1e-8)
