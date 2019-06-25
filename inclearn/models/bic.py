import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import calibration
from inclearn.models.icarl import ICarl

EPSILON = 1e-8


class BiC(ICarl):
    """Implements Large Scale Incremental Learning.

    * https://arxiv.org/abs/1905.13260
    """

    def __init__(self, args):
        if args["validation"] <= 0.:
            raise Exception("BiC needs validation data!")

        self._temperature = args["temperature"]

        super().__init__(args)

    def _after_task(self, inc_dataset):
        super()._after_task(inc_dataset)

        if self._task == 0:
            return

        _, val_loader = inc_dataset.get_custom_loader(
            list(range(0, self._n_classes)), mode="test", data_source="val"
        )

        print("Compute bias correction.")

        self._bic = calibration.calibrate(
            self._network, val_loader, self._device,
            indexes=[(self._n_classes - self._task_size, self._n_classes)],
            calibration_type="linear"
        ).to(self._device)

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for inputs, targets in loader:
            logits = self._network(inputs.to(self._device))
            if self._task > 0:
                logits = self._bic(logits)

            ytrue.append(targets.numpy())
            ypred.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _compute_loss(self, inputs, logits, targets, onehot_targets):
        loss = F.cross_entropy(logits, targets)

        if self._old_model is not None:
            old_targets = self._old_model(inputs).detach()

            loss += F.binary_cross_entropy_with_logits(
                logits[..., :-self._task_size] / self._temperature,
                torch.sigmoid(old_targets / self._temperature)
            )

        return loss
