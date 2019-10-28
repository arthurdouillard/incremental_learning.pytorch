import collections
import copy
import logging
import os
import pdb

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, losses, network, schedulers, utils
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._old_device = args["device"][-1]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", "icarl")
        self._n_classes = 0

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._save_model = args["save_model"]

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._herding_indexes = []

        self._epoch_metrics = collections.defaultdict(list)

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network_path):
        if self._network is not None:
            del self._network

        logger.info("Loading model from {}.".format(network_path))
        self._network = torch.load(network_path)
        self._network.to(self._device)
        self._network.device = self._device
        self._network.classifier.device = self._device

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

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
                logger.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

    def _training_step(self, train_loader, val_loader, initial_epoch, nb_epochs):
        best_epoch, best_acc = -1, -1.
        wait = 0

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                self._optimizer.zero_grad()
                loss = self._forward_loss(inputs, targets, memory_flags)
                loss.backward()
                self._optimizer.step()

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def _forward_loss(self, inputs, targets, memory_flags):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        if self._random_noise_config:
            random_noise = torch.randn(self._random_noise_config["nb_per_batch"], *inputs.shape[1:])
            inputs = torch.cat((inputs, random_noise.to(self._device)))

        logits = self._network(inputs)

        loss = self._compute_loss(inputs, logits, targets, onehot_targets, memory_flags)

        if not utils._check_loss(loss):
            pdb.set_trace()

        self._metrics["loss"] += loss.item()

        return loss

    def _after_task(self, inc_dataset):
        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

        self._old_model = self._network.copy().freeze().to(self._old_device)

        self._network.on_task_end()
        self.plot_tsne()

    def plot_tsne(self):
        if self.folder_result:
            loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())[1]
            embeddings, targets = utils.extract_features(self._network, loader)
            utils.plot_tsne(
                os.path.join(self.folder_result, "tsne_{}".format(self._task)), embeddings, targets
            )

    def _eval_task(self, data_loader):
        ypred, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets, onehot_targets, memory_flags):
        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs.to(self._old_device)).detach()
                                       ).to(self._device)

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self._network(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
                " Have you forgotten to call `before_task`?"
            )
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes)
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._network.extract(inputs).detach()
            preds = self._get_closest(self._means, F.normalize(features))

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((100, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection == "random":
                    selected_indexes = np.random.permutation(len(features))[:memory_per_class]
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            selected_indexes = herding_indexes[class_idx][:memory_per_class]
            herding_indexes[class_idx] = selected_indexes

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        features, targets_ = utils.extract_features(model, loader)

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return np.argsort(score_icarl, axis=1)[:, -1], targets_
