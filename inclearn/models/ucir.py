import math

import torch
from torch.nn import functional as F

from inclearn.lib import network
from inclearn.models.icarl import ICarl


class UCIR(ICarl):
    """Implements Learning a Unified Classifier Incrementally via Rebalancing

    * http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
    """

    def __init__(self, args):
        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._n_classes = 0

        self._network = network.BasicNet(
            args["convnet"], device=self._device, cosine_similarity=True, scaling_factor=True,
            return_features=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._lambda = 5

        self._herding_indexes = []

    def _train_task(self, *args, **kwargs):
        for p in self._network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        super()._train_task(*args, **kwargs)

    def _compute_loss(self, inputs, features_logits, targets, onehot_targets):
        features, logits = features_logits

        clf_loss = F.cross_entropy(self._network.post_process(logits), targets)
        if self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits = self._old_model(inputs)

            distil_loss = torch.mean(1 - F.cosine_similarity(features, old_features))
            #distil_loss *= self._lambda * math.sqrt(
            #    self._task_size / (self._n_classes - self._task_size)
            #)
        else:
            distil_loss = torch.tensor(0.)

        return clf_loss + distil_loss
