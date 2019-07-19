import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import calibration, herding, utils
from inclearn.models.icarl import ICarl

EPSILON = 1e-8


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

        super().__init__(args)

    def _after_task(self, inc_dataset):
        self._train_memory = self.build_examplars(
            inc_dataset,
            data_source="train",
            quantity=int(self._memory_per_class * (1 - self._validation))
        )
        self._val_memory = self.build_examplars(
            inc_dataset, data_source="val", quantity=int(self._memory_per_class * self._validation)
        )

        self._old_model = self._network.copy().freeze()

        if self._task == 0:
            return

        _, val_loader = inc_dataset.get_custom_loader(
            list(range(self._n_classes - self._task_size, self._n_classes)),
            mode="test",
            data_source="val",
            memory=self._val_memory
        )

        print("Compute bias correction.")

        self._bic = calibration.calibrate(
            self._network,
            val_loader,
            self._device,
            indexes=[(self._n_classes - self._task_size, self._n_classes)],
            calibration_type="linear"
        ).to(self._device)

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for inputs, targets, _ in loader:
            logits = self._network(inputs.to(self._device))
            if self._task > 0:
                logits = self._bic(logits)

            ytrue.append(targets.numpy())
            ypred.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _compute_loss(self, inputs, logits, targets, onehot_targets, memory_flags):
        loss = F.cross_entropy(logits, targets)

        if self._old_model is not None:
            old_targets = self._old_model(inputs).detach()

            loss += F.binary_cross_entropy_with_logits(
                logits[..., :-self._task_size] / self._temperature,
                torch.sigmoid(old_targets / self._temperature)
            )

        return loss

    def build_examplars(self, inc_dataset, data_source, quantity):
        print("Building & updating memory.")

        if data_source == "train":
            herding_indexes = self._herding_indexes
        else:
            herding_indexes = self._herding_val_indexes

        data_memory, targets_memory = [], []

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                herding_indexes.append(herding.icarl_selection(features, quantity))

            # Reducing examplars:
            selected_indexes = herding_indexes[class_idx][:quantity]
            herding_indexes[class_idx] = selected_indexes

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

        return np.concatenate(data_memory), np.concatenate(targets_memory)

    def get_val_memory(self):
        return self._val_memory

    def get_memory(self):
        return self._train_memory
