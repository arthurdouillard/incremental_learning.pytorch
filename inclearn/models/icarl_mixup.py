import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, network, utils
from inclearn.models.icarl import ICarl, compute_examplar_mean, create_alph

EPSILON = 1e-8


class ICarlMixUp(ICarl):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__(args)

    def get_memory(self):
        x, y = self._data_memory, self._targets_memory
        y = np.eye(self._n_classes)[y]

        indexes = np.random.permutation(x.shape[0])

        x_r, y_r = None, None
        if x.shape[0] % 2 != 0:
            excedent = x.shape[0] % 2
            indexes = indexes[excedent:]

            x_r = x[indexes[:excedent]]
            y_r = y[indexes[:excedent]]

        x, y = x[indexes], y[indexes]

        assert x.shape[0] % 2 == 0, x.shape[0]
        x1, x2 = x[x.shape[0]//2:], x[:x.shape[0]//2]
        y1, y2 = y[x.shape[0]//2:], y[:x.shape[0]//2]

        alpha = 0.6
        lam = np.random.beta(alpha, alpha, size=x.shape[0] // 2)
        lam_x = lam.reshape(-1, 1, 1, 1)
        lam_y = lam.reshape(-1, 1)

        m_x = lam_x * x1 + (1 - lam_x) * x2
        m_y = lam_y * y1 + (1 - lam_y) * y2

        print(m_x.shape[0], self._memory_per_class * self._n_classes)

        if x_r is not None:
            m_x = np.concatenate((m_x, x_r))
            m_y = np.concatenate((m_y, y_r))

        print("Memory of size {} get reduced to {}.".format(x.shape[0], m_x.shape[0]))
        return m_x, m_y

    def _forward_loss(self, inputs, targets):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        logits = self._network(inputs)

        return self._compute_loss(inputs, logits.double(), targets.double())

    def _compute_loss(self, inputs, logits, onehot_targets):
        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        return loss

    def build_examplars(self, inc_dataset):
        print("Building & updating memory.")

        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim))

        for class_idx in range(self._n_classes):
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test")
            features, targets = utils.extract_features(
                self._network, loader
            )
            features_flipped, _ = utils.extract_features(
                self._network, inc_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                her = herding.select_examplars(
                    features, self._memory_per_class * 2
                )
                self._herding_matrix.append(her)

            examplar_mean = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )

            alph = create_alph(self._herding_matrix[class_idx], self._memory_per_class * 2)
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)
