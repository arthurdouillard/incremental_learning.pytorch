import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import data, factory, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class PODNet(ICarl):
    """Pooled Output Distillation Network.

    # Reference:
        * Small Task Incremental Learning
          Douillard et al. 2020
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
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        classifier_kwargs = args.get("classifier_config", {})
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
            gradcam_hook=bool(self._gradcam_distil)
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []

        self._weight_generation = args.get("weight_generation")

        self._meta_transfer = args.get("meta_transfer", {})
        if self._meta_transfer:
            assert "mtl" in args["convnet"]

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
        if self._meta_transfer:
            logger.info("Setting task meta-transfer")
            self.set_meta_transfer()

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug("nb {}.".format(len(train_loader.dataset)))

        if self._meta_transfer.get("clip"):
            logger.info(f"Clipping MTL weights ({self._meta_transfer.get('clip')}).")
            clipper = BoundClipper(*self._meta_transfer.get("clip"))
        else:
            clipper = None
        self._training_step(
            train_loader, val_loader, 0, self._n_epochs, record_bn=True, clipper=clipper
        )

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")
            if self._finetuning_config["scaling"]:
                logger.info(
                    "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
                )
                self._post_processing_type = self._finetuning_config["scaling"]

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
                parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
            )
            self._scheduler = None
            self._training_step(
                loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"],
                record_bn=False
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
        if self._gradcam_distil:
            self._network.zero_grad()
            self._network.unset_gradcam_hook()
            self._old_model = self._network.copy().eval().to(self._device)
            self._network.on_task_end()

            self._network.set_gradcam_hook()
            self._old_model.set_gradcam_hook()
        else:
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
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
        else:
            raise ValueError(self._evaluation_type)

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
        elif self._groupwise_factors == "ucir":
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                },
                {
                    "params": self._network.classifier.new_weights,
                    "lr": self._lr
                },
            ]
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

        if self._class_weights_config:
            self._class_weights = torch.tensor(
                data.get_class_weights(train_loader.dataset, **self._class_weights_config)
            ).to(self._device)
        else:
            self._class_weights = None

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                old_features = old_outputs["raw_features"]
                old_atts = old_outputs["attention"]

        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss = losses.nca(
                logits,
                targets,
                memory_flags=memory_flags,
                class_weights=self._class_weights,
                **nca_config
            )
            self._metrics["nca"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(scaled_logits, targets)
            self._metrics["cce"] += loss.item()

        # --------------------
        # Distillation losses:
        # --------------------

        if self._old_model is not None:
            if self._pod_flat_config:
                if self._pod_flat_config["scheduled_factor"]:
                    factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_flat_config.get("factor", 1.)

                pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
                loss += pod_flat_loss
                self._metrics["flat"] += pod_flat_loss.item()

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                pod_spatial_loss = factor * losses.pod(
                    old_atts,
                    atts,
                    memory_flags=memory_flags.bool(),
                    task_percent=(self._task + 1) / self._n_tasks,
                    **self._pod_spatial_config
                )
                loss += pod_spatial_loss
                self._metrics["pod"] += pod_spatial_loss.item()

            if self._perceptual_features:
                percep_feat = losses.perceptual_features_reconstruction(
                    old_atts, atts, **self._perceptual_features
                )
                loss += percep_feat
                self._metrics["p_feat"] += percep_feat.item()

            if self._perceptual_style:
                percep_style = losses.perceptual_style_reconstruction(
                    old_atts, atts, **self._perceptual_style
                )
                loss += percep_style
                self._metrics["p_sty"] += percep_style.item()

            if self._gradcam_distil:
                top_logits_indexes = logits[..., :-self._task_size].argmax(dim=1)
                try:
                    onehot_top_logits = utils.to_onehot(
                        top_logits_indexes, self._n_classes - self._task_size
                    ).to(self._device)
                except:
                    import pdb
                    pbd.set_trace()

                old_logits = old_outputs["logits"]

                logits[
                    ..., :-self._task_size].backward(gradient=onehot_top_logits, retain_graph=True)
                old_logits.backward(gradient=onehot_top_logits)

                if len(outputs["gradcam_gradients"]) > 1:
                    gradcam_gradients = torch.cat(
                        [g.to(self._device) for g in outputs["gradcam_gradients"] if g is not None]
                    )
                    gradcam_activations = torch.cat(
                        [
                            a.to(self._device)
                            for a in outputs["gradcam_activations"]
                            if a is not None
                        ]
                    )
                else:
                    gradcam_gradients = outputs["gradcam_gradients"][0]
                    gradcam_activations = outputs["gradcam_activations"][0]

                if self._gradcam_distil.get("scheduled_factor", False):
                    factor = self._gradcam_distil["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._gradcam_distil.get("factor", 1.)

                try:
                    attention_loss = factor * losses.gradcam_distillation(
                        gradcam_gradients, old_outputs["gradcam_gradients"][0].detach(),
                        gradcam_activations, old_outputs["gradcam_activations"][0].detach()
                    )
                except:
                    import pdb
                    pdb.set_trace()

                self._metrics["grad"] += attention_loss.item()
                loss += attention_loss

                self._old_model.zero_grad()
                self._network.zero_grad()

        return loss


class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)
