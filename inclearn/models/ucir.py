import math
import warnings

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, losses, network, utils
from inclearn.models.icarl import ICarl


class UCIR(ICarl):
    """Implements Learning a Unified Classifier Incrementally via Rebalancing

    * http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]
        self._herding_selection = args.get("herding_selection", "icarl")

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        if "distillation_loss" in args:
            warnings.warn("distillation_loss is replaced by less_forget")
            args["less_forget"] = args["distillation_loss"]
            del args["distillation_loss"]

        self._use_mimic_score = args.get("mimic_score", False)
        self._use_less_forget = args.get("less_forget", True)
        self._lambda_schedule = args.get("lambda_schedule", True)
        self._use_ranking = args.get("ranking_loss", True)

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

        self._warmup_config = args.get("warmup")
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._lambda = args.get("base_lambda", 5)
        self._nb_negatives = args.get("nb_negatives", 2)
        self._margin = args.get("ranking_margin", 0.2)

        self._weight_generation = args.get("weight_generation")

        self._herding_indexes = []

        self._eval_type = args.get("eval_type", "nme")

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
            s = self._network.post_processor.factor

        print("Scale is {}.".format(s))
        self._args["_logs"]["scale"].append(s)

        super()._after_task(inc_dataset)

    def _eval_task(self, data_loader):
        if self._eval_type == "nme":
            return super()._eval_task(data_loader)
        elif self._eval_type == "cnn":
            ypred = []
            ytrue = []

            for inputs, targets, _ in data_loader:
                ytrue.append(targets.numpy())

                inputs = inputs.to(self._device)
                preds = F.softmax(self._network(inputs)[1], dim=1).argmax(dim=1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._eval_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network,
                self._weight_generation if self._task == 0 else "basic",
                self._n_classes, self._task_size,
                self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        self._gen_weights()

        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                print("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, *args, **kwargs):
        for p in self._network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        super()._train_task(*args, **kwargs)

    def _compute_loss(self, inputs, features_logits, targets, onehot_targets, memory_flags):
        features, logits = features_logits

        # Classification loss is cosine + learned factor + softmax:
        loss = F.cross_entropy(self._network.post_process(logits), targets)
        self._metrics["clf"] += loss.item()

        if self._old_model is not None:
            with torch.no_grad():
                old_features, old_logits = self._old_model(inputs)

            if self._use_less_forget:
                if self._lambda_schedule:
                    scheduled_lambda = self._lambda * math.sqrt(self._n_classes / self._task_size)
                else:
                    scheduled_lambda = 1.

                lessforget_loss = scheduled_lambda * losses.embeddings_similarity(
                    old_features, features
                )
                loss += lessforget_loss
                self._metrics["lessforget"] += lessforget_loss.item()
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
                    nb_negatives=self._nb_negatives,
                    margin=self._margin
                )
                loss += ranking_loss
                self._metrics["rank"] += ranking_loss.item()

        return loss
