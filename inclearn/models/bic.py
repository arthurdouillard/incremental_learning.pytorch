import logging

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import calibration, herding, losses, utils
from inclearn.models.icarl import ICarl

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class BiC(ICarl):
    """Implements Large Scale Incremental Learning.

    * https://arxiv.org/abs/1905.13260
    """

    def __init__(self, args):
        if args["validation"] <= 0.:
            raise Exception("BiC needs validation data!")

        self._validation = args["validation"]
        self._temperature = args["temperature"]
        self._herding_val_indexes = []
        logger.info("Initializing BiC")

        super().__init__(args)

    def _after_task(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            self.inc_dataset,
            self._herding_indexes,
            memory_per_class=int(self._memory_per_class * (1 - self._validation))
        )

        self._old_model = self._network.copy().freeze()

        val_x, val_y, self._herding_val_indexes, _ = self.build_examplars(
            self.inc_dataset,
            self._herding_val_indexes,
            data_source="val",
            memory_per_class=int(self._memory_per_class * self._validation)
        )
        self._val_memory = (val_x, val_y)

        if self._task == 0:
            return

        _, val_loader = inc_dataset.get_custom_loader(
            [], mode="test", data_source="val", memory=self._val_memory
        )

        print("Compute bias correction.")

        self._bic = calibration.calibrate(
            self._network,
            val_loader,
            self._device,
            indexes=[(self._n_classes - self._task_size, self._n_classes)],
            calibration_type="linear"
        ).to(self._device)
        self._old_model.post_processor = self._bic

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for input_dict in loader:
            logits = self._network(input_dict["inputs"].to(self._device))
            if self._task > 0:
                logits = self._bic(logits)

            ytrue.append(input_dict["targets"].numpy())
            ypred.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _compute_loss(self, inputs, logits, targets, onehot_targets, memory_flags):
        loss = F.cross_entropy(logits, targets)

        if self._old_model is not None:
            old_targets = self._old_model.post_process(self._old_model(inputs)).detach()

            loss += F.binary_cross_entropy_with_logits(
                logits[..., :-self._task_size] / self._temperature,
                torch.sigmoid(old_targets / self._temperature)
            )

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    def get_val_memory(self):
        return self._val_memory

    def get_memory(self):
        return self._data_memory, self._targets_memory
