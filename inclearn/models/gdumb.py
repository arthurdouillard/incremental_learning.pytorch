import logging
import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import data, factory, losses, network, utils
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class GDumb(ICarl):
    """

    # Reference:
        * GDumb: A Simple Approach that Questions Our Progress in Continual Learning
          Prabhu et al. ECCV 2020
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._first_task_n_epochs = args["first_task_epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = {"type": "first"}
        self._n_classes = 0
        self._nb_inc_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._evaluation_type = "cnn"

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs={
                "type": "fc",
                "use_bias": True
            },
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=False,
        )

        self._cutmix_alpha = args.get("cutmix_alpha", 1.0)
        self._cutmix_prob = args.get("cutmix_prob", 0.5)

        self._grad_clip = args.get("grad_clip", 10.0)

        self._examplars = {}
        self._means = None

        self._old_model = None
        self._first_model = None

        self._herding_indexes = []

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):
        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -self._grad_clip, self._grad_clip))

        if self._task > 0:
            train_loader = self.inc_dataset.get_custom_loader(
                [], memory=self.get_memory(), mode="train"
            )[1]

        self._training_step(
            train_loader, val_loader, 0,
            self._first_task_n_epochs if self._task == 0 else self._n_epochs
        )

    @property
    def weight_decay(self):
        return self._weight_decay

    def _after_task(self, inc_dataset):
        if self._task == 0:
            self._first_model = self._network.copy().freeze().to(self._device)
            self._first_model.on_task_end()

    def _before_task(self, train_loader, val_loader):
        if self._task != 0:
            self._network = self._first_model.copy()
            self._nb_inc_classes += self._task_size
            utils.add_new_weights(
                self._network, "basic", self._n_classes, self._nb_inc_classes, self.inc_dataset
            )
        else:
            utils.add_new_weights(
                self._network, "basic", self._n_classes, self._task_size, self.inc_dataset
            )
        self._network.classifier.reset_weights()

        self._n_classes += self._task_size
        logger.info("Now {} examplars per class.".format(self._memory_per_class))
        logger.info(f"Nb classes in classifier {len(self._network.classifier.weights)}")

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(params, self._opt_name, self._lr, self.weight_decay)

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

    def _forward_loss(
        self,
        training_network,
        inputs,
        targets,
        memory_flags,
        gradcam_grad=None,
        gradcam_act=None,
        **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        is_cutmix_applied = False
        if np.random.rand(1) < self._cutmix_prob:
            is_cutmix_applied = True
            inputs, targets_a, targets_b, lam = apply_cutmix(
                inputs, targets, alpha=self._cutmix_alpha
            )

        outputs = training_network(inputs)
        logits = outputs["logits"]

        if is_cutmix_applied:
            loss = lam * F.cross_entropy(logits, targets_a
                                        ) + (1 - lam) * F.cross_entropy(logits, targets_b)
        else:
            loss = F.cross_entropy(logits, targets)

        self._metrics["cce"] += loss.item()

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss


def apply_cutmix(x, y, alpha=1.0):
    assert (alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
