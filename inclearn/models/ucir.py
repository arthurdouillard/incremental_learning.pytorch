import logging
import math
import warnings

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, losses, network, utils
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class UCIR(ICarl):
    """Implements Learning a Unified Classifier Incrementally via Rebalancing

    * http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")

        self._use_mimic_score = args.get("mimic_score")
        self._use_less_forget = args.get("less_forget")
        self._lambda_schedule = args.get("lambda_schedule", True)
        self._use_ranking = args.get("ranking_loss")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {}),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True,
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._lambda = args.get("base_lambda", 5)
        self._nb_negatives = args.get("nb_negatives", 2)
        self._margin = args.get("ranking_margin", 0.2)

        self._weight_generation = args.get("weight_generation")

        self._herding_indexes = []

        self._eval_type = args.get("eval_type", "nme")

        self._meta_transfer = args.get("meta_transfer", False)
        if self._meta_transfer:
            assert args["convnet"] == "rebuffi_mtl"

        self._args = args
        self._args["_logs"] = {}

    def _after_task(self, inc_dataset):
        if "scale" not in self._args["_logs"]:
            self._args["_logs"]["scale"] = []

        if self._network.post_processor is None:
            s = None
        elif hasattr(self._network.post_processor, "scale"):
            s = self._network.post_processor.scale.item()
        elif hasattr(self._network.post_processor, "factor"):
            s = self._network.post_processor.factor.item()

        print("Scale is {}.".format(s))
        self._args["_logs"]["scale"].append(s)

        super()._after_task(inc_dataset)

    def _eval_task(self, data_loader):
        if self._eval_type == "nme":
            return super()._eval_task(data_loader)
        elif self._eval_type == "cnn":
            ypred = []
            ytrue = []

            for input_dict in data_loader:
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
            raise ValueError(self._eval_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        self._gen_weights()

        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling, self._optimizer, self._n_epochs, lr_decay=self._lr_decay
        )

    def _train_task(self, train_loader, val_loader):
        if self._meta_transfer:
            logger.info("Setting task meta-transfer")
            self.set_meta_transfer()

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        self._training_step(train_loader, val_loader, 0, self._n_epochs)

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")

            self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                self.inc_dataset, self._herding_indexes
            )
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())

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
                loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"],
                record_bn=False
            )

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        features, logits = outputs["raw_features"], outputs["logits"]

        # Classification loss is cosine + learned factor + softmax:
        loss = F.cross_entropy(self._network.post_process(logits), targets)
        self._metrics["clf"] += loss.item()

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                old_features = old_outputs["raw_features"]

            if self._use_less_forget:
                if self._lambda_schedule:
                    scheduled_lambda = self._lambda * math.sqrt(self._n_classes / self._task_size)
                else:
                    scheduled_lambda = 1.

                lessforget_loss = scheduled_lambda * losses.embeddings_similarity(
                    old_features, features
                )
                loss += lessforget_loss
                self._metrics["lf"] += lessforget_loss.item()
            elif self._use_mimic_score:
                old_class_logits = logits[..., :self._n_classes - self._task_size]
                old_class_old_logits = old_logits[..., :self._n_classes - self._task_size]

                mimic_loss = F.mse_loss(old_class_logits, old_class_old_logits)
                mimic_loss *= (self._n_classes - self._task_size)
                loss += mimic_loss
                self._metrics["mimic"] += mimic_loss.item()

            if self._use_ranking:
                ranking_loss = losses.ucir_ranking(
                    logits,
                    targets,
                    self._n_classes,
                    self._task_size,
                    nb_negatives=max(self._nb_negatives, self._task_size),
                    margin=self._margin
                )
                loss += ranking_loss
                self._metrics["rank"] += ranking_loss.item()

        return loss
