import copy
import logging
import math

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from inclearn.lib import data, factory, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class ZIL2(ICarl):

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._less_forget_config = args.get("less_forget", {})
        self._lambda_schedule = args.get("lambda_schedule", False)

        self._ams_config = args.get("adaptative_margin_softmax", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._attention_residual_config = args.get("attention_residual", {})

        self._groupwise_factors = args.get("groupwise_factors", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {}),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._weight_generation = args.get("weight_generation")

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._args = args
        self._args["_logs"] = {}

    def _train_task(self, train_loader, val_loader):
        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug("nb {}.".format(len(train_loader.dataset)))
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")

            if self._finetuning_config["sampling"] == "undersampling":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())
            elif self._finetuning_config["sampling"] == "oversampling":
                _, loader = self.inc_dataset.get_custom_loader(
                    list(range(self._n_classes - self._task_size, self._n_classes)),
                    memory=self.get_memory(),
                    mode="train",
                    sampler=samplers.MemoryOverSampler
                )

            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.parameters()
            elif self._finetuning_config["tuning"] == "classifier_scale":
                parameters = [
                    {
                        "params": self._network.classifier.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }, {
                        "params": self._network.post_processor.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }
                ]
            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], self._weight_decay
            )
            self._scheduler = None
            self._training_step(
                loader, val_loader, self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"]
            )

    def _after_task(self, inc_dataset):
        self._monitor_scale()
        super()._after_task(inc_dataset)

    def _monitor_scale(self):
        if "scale" not in self._args["_logs"]:
            self._args["_logs"]["scale"] = []

        if self._network.post_processor is None:
            s = None
        elif hasattr(self._network.post_processor, "factor"):
            if isinstance(self._network.post_processor.factor, float):
                s = self._network.post_processor.factor
            else:
                s = self._network.post_processor.factor.item()
        else:
            s = None
        logger.info("Scale is {}.".format(s))

        self._args["_logs"]["scale"].append(s)

    def _eval_task(self, test_loader):
        ypred = []
        ytrue = []

        for input_dict in test_loader:
            ytrue.append(input_dict["targets"].numpy())

            inputs = input_dict["inputs"].to(self._device)
            logits = self._network(inputs)["logits"].detach()

            preds = F.softmax(logits, dim=-1)
            ypred.append(preds.cpu().numpy())

        ypred = np.concatenate(ypred)
        ytrue = np.concatenate(ytrue)

        self._last_results = (ypred, ytrue)

        return ypred, ytrue

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        self._gen_weights()
        self._n_classes += self._task_size
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None:
                    continue
                params.append(
                    {
                        "params": group_params,
                        "lr": self._lr * self._groupwise_factors.get(group_name, 1.0)
                    }
                )
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(
            params, self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]
        scaled_logits = self._network.post_process(logits)

        if self._old_model is not None:
            self._old_model.zero_grad()
            old_outputs = self._old_model(inputs)
            old_features = old_outputs["raw_features"].detach()
            old_atts = [a.detach() for a in old_outputs["attention"]]

        if self._ams_config:
            ams_config = copy.deepcopy(self._ams_config)
            if self._network.post_processor:
                ams_config["scale"] = self._network.post_processor.factor

            loss = losses.additive_margin_softmax_ce(logits, targets, **ams_config)
            self._metrics["ams"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(scaled_logits, targets)
            self._metrics["cce"] += loss.item()

        # ----------------------
        # Regularization losses:
        # ----------------------

        # --------------------
        # Distillation losses:
        # --------------------

        if self._old_model is not None:
            if self._less_forget_config:
                if self._less_forget_config["scheduled_factor"]:
                    factor = self._less_forget_config["lambda"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._less_forget_config.get("factor", 1.)

                distil_loss = factor * losses.embeddings_similarity(old_features, features)
                loss += distil_loss
                self._metrics["lf"] += distil_loss.item()

            if self._attention_residual_config:
                if self._attention_residual_config.get("scheduled_factor", False):
                    factor = self._attention_residual_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._attention_residual_config.get("factor", 1.)

                attention_loss = factor * losses.residual_attention_distillation(
                    old_atts,
                    atts,
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._attention_residual_config
                )
                loss += attention_loss
                self._metrics["att"] += attention_loss.item()

        return loss
