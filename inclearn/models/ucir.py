import math

import torch
from torch.nn import functional as F

from inclearn.lib import factory, network
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
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        self._use_distil = args.get("distillation_loss", True)
        self._lambda_schedule = args.get("lambda_schedule", True)
        self._use_ranking = args.get("ranking_loss", True)
        self._scaling_factor = args.get("scaling_factor", True)

        self._network = network.BasicNet(
            args["convnet"],
            device=self._device,
            cosine_similarity=True,
            scaling_factor=self._scaling_factor,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._lambda = args.get("base_lambda", 5)
        self._nb_negatives = args.get("nb_negatives", 2)
        self._margin = args.get("ranking_margin", 0.2)
        self._use_imprinted_weights = args.get("imprinted_weights", True)

        self._herding_indexes = []

    def _before_task(self, train_loader, val_loader):
        if self._use_imprinted_weights:
            self._network.add_imprinted_classes(
                list(range(self._n_classes, self._n_classes + self._task_size)),
                self.inc_dataset)
        else:
            self._network.add_classes(self._task_size)
        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

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
            if self._lambda_schedule:
                scheduled_lambda = self._lambda * math.sqrt(
                    self._n_classes / self._task_size
                )
            else:
                scheduled_lambda = 1.

            distil_loss = scheduled_lambda * F.cosine_embedding_loss(
                features, old_features,
                torch.ones(inputs.shape[0]).to(self._device)
            )

            # Ranking loss maximizing the inter-class separation between old & new:

            # 1. Fetching from the batch only samples from the batch that belongs
            #    to old classes:
            old_indexes = targets.lt(self._n_classes - 1)
            old_logits = logits[old_indexes]
            old_targets = targets[old_indexes]

            # 2. Getting positive values, aka ground-truth's logit predictions:
            old_values = old_logits[torch.arange(len(old_logits)), old_targets]
            old_values = old_values.repeat(self._nb_negatives, 1).t().contiguous().view(-1)

            # 3. Getting top-k negative values:
            nb_old_classes = self._n_classes - self._task_size
            negative_indexes = old_logits[..., nb_old_classes:].argsort(dim=1, descending=True)[..., :self._nb_negatives] + nb_old_classes
            new_values = old_logits[torch.arange(len(old_logits)).view(-1, 1), negative_indexes].view(-1)

            ranking_loss = F.margin_ranking_loss(
                old_values,
                new_values,
                -torch.ones(len(old_values)).to(self._device),
                margin=self._margin
            )
        else:
            distil_loss = torch.tensor(0.)
            ranking_loss = torch.tensor(0.)

        loss = clf_loss
        if self._use_distil:
            loss += distil_loss
        if self._use_ranking:
            loss += ranking_loss

        return loss
