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
            args["convnet"],
            #convnet_kwargs={"last_relu": False},
            device=self._device,
            cosine_similarity=True,
            scaling_factor=True,
            return_features=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._lambda = 5
        self._nb_negatives = 2
        self._margin = 0.2

        self._herding_indexes = []

    @property
    def _memory_per_class(self):
        return 20  #self._memory_size // self._n_classes

    def _train_task(self, *args, **kwargs):
        for p in self._network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        super()._train_task(*args, **kwargs)

    def _compute_loss(self, inputs, features_logits, targets, onehot_targets, memory_flags):
        features, logits = features_logits

        # Classification loss is cosine + learned factor + softmax:
        clf_loss = F.cross_entropy(self._network.post_process(logits), targets)

        if self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits = self._old_model(inputs)

            # Distillation loss fixing the deviation problem:
            scheduled_lambda = self._lambda * math.sqrt(
                self._task_size / (self._n_classes - self._task_size)
            ) / logits.shape[0]

            distil_loss = scheduled_lambda * F.cosine_embedding_loss(
                features, old_features,
                torch.ones(inputs.shape[0]).to(self._device)
            )

            # Ranking loss maximizing the inter-class separation between old & new:
            highest_confidence_indexes = logits.argsort(dim=1, descending=True)
            old_indexes = targets.lt(self._n_classes - 1)
            highest_confidence_indexes[old_indexes]


            ranking_loss = torch.tensor(0.)
            for batch_index in range(logits.shape[0]):
                if not memory_flags[batch_index]:
                    # Is new class
                    continue

                class_index = 0
                counter = 0

                while counter < self._nb_negatives:
                    if highest_confidence_indexes[batch_index, class_index] > (
                        self._n_classes - self._task_size
                    ):
                        class_index += 1
                        # Is old class, but we only want new class as positive.
                        continue

                    positive = logits[batch_index, targets[batch_index]]
                    negative = logits[batch_index, class_index]

                    ranking_loss += max(self._margin - positive + negative, 0)
                    counter += 1
                    class_index += 1

            if memory_flags.to(self._device).sum().item() > 1:
                ranking_loss *= (1 / memory_flags.to(self._device).sum()).float()
                distil_loss += ranking_loss
            # else there was no memory data in this batch.
        else:
            distil_loss = torch.tensor(0.)
            ranking_loss = torch.tensor(0.)

        return clf_loss + distil_loss + ranking_loss
