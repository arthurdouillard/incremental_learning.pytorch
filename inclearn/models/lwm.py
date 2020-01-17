import logging
import pdb

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, loops, losses, network, utils
from inclearn.models import IncrementalLearner

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class LwM(IncrementalLearner):

    def __init__(self, args):
        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]

        self._distillation_config = args["distillation_config"]
        self._attention_config = args.get("attention_config", {})

        logger.info("Initializing LwM")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            gradcam_hook=True
        )

        self._n_classes = 0
        self._old_model = None

    def _before_task(self, data_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )
        if self._scheduling is None:
            self._scheduler = None
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self._optimizer, self._scheduling, gamma=self._lr_decay
            )

    def _train_task(self, train_loader, val_loader):
        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self._n_epochs,
            self._optimizer,
            scheduler=self._scheduler,
            train_function=self._forward_loss,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks
        )

    def _after_task(self, inc_dataset):
        self._network.zero_grad()
        self._network.unset_gradcam_hook()
        self._old_model = self._network.copy().eval().to(self._device)
        self._network.on_task_end()

        self._network.set_gradcam_hook()
        self._old_model.set_gradcam_hook()

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

    def _forward_loss(
        self,
        training_network,
        inputs,
        targets,
        memory_flags,
        metrics,
        gradcam_grad=None,
        gradcam_act=None,
        **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags, metrics)

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, metrics):
        logits = outputs["logits"]

        if self._old_model is None:
            # Classification loss
            loss = F.cross_entropy(logits, targets)
            metrics["clf"] += loss.item()
        else:
            self._old_model.zero_grad()
            old_outputs = self._old_model(inputs)
            old_logits = old_outputs["logits"]

            # Classification loss
            loss = F.cross_entropy(
                logits[..., -self._task_size:], (targets - self._n_classes + self._task_size)
            )
            metrics["clf"] += loss.item()

            # Distillation on probabilities
            distill_loss = self._distillation_config["factor"] * F.binary_cross_entropy_with_logits(
                logits[..., :-self._task_size], torch.sigmoid(old_logits.detach())
            )
            metrics["dis"] += distill_loss.item()
            loss += distill_loss

            # Distillation on gradcam-generated attentions
            if self._attention_config:
                top_logits_indexes = logits[..., :-self._task_size].argmax(dim=1)
                onehot_top_logits = utils.to_onehot(
                    top_logits_indexes, self._n_classes - self._task_size
                ).to(self._device)

                logits[..., :-self._task_size].backward(
                    gradient=onehot_top_logits, retain_graph=True
                )
                old_logits.backward(gradient=onehot_top_logits)

                if len(outputs["gradcam_gradients"]) > 1:
                    gradcam_gradients = torch.cat(
                        [g.to(self._device) for g in outputs["gradcam_gradients"]]
                    )
                    gradcam_activations = torch.cat(
                        [a.to(self._device) for a in outputs["gradcam_activations"]]
                    )
                else:
                    gradcam_gradients = outputs["gradcam_gradients"][0]
                    gradcam_activations = outputs["gradcam_activations"][0]

                attention_loss = losses.gradcam_distillation(
                    gradcam_gradients, old_outputs["gradcam_gradients"][0].detach(),
                    gradcam_activations, old_outputs["gradcam_activations"][0].detach(),
                    **self._attention_config
                )
                metrics["ad"] += attention_loss.item()
                loss += attention_loss

                self._old_model.zero_grad()
                self._network.zero_grad()

        return loss
