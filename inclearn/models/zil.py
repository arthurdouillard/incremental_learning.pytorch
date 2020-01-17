import copy
import functools
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from inclearn.lib import data, distance, factory, loops, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.lib.network.word import Word2vec
from inclearn.models.base import IncrementalLearner

logger = logging.getLogger(__name__)


class ZIL(IncrementalLearner):

    def __init__(self, args):
        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]

        self._generate_unseen = args.get("generate_unseen", {})

        # Losses definition
        self._gmm_config_loss = args.get("gmm_config", {})

        logger.info("Initializing ZIL")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {"type": "cosine"}),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            wordembeddings_kwargs=args["word_embeddings"],
            device=self._device,
            extract_no_act=True,
            classifier_no_act=True
        )
        self._word_embeddings_kwargs = args["word_embeddings"]

        self._n_classes = 0
        self._old_model = None

    def get_class_label(self, fake_class_ids):
        return torch.tensor([self.inc_dataset.class_order[0][i] for i in fake_class_ids])

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        utils.add_new_weights(
            self._network, {"type": "imprinted"} if self._task > 1 else {"type": "basic"},
            self._n_classes, self._task_size, self.inc_dataset
        )

        self._optimizer = factory.get_optimizer(
            [
                {
                    "params": self._network.classifier.parameters(),
                    "lr": self._lr
                }, {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                }, {
                    "params": self._network.post_processor.parameters(),
                    "lr": self._lr
                }
            ], self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
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
        if self._gmm_config_loss:
            self._network.word_embeddings = Word2vec(
                **self._word_embeddings_kwargs, device=self._device
            )
            self._train_gmm()

        self._network.on_task_end()
        self._old_model = self._network.copy().eval().to(self._device)

    def _gmm_loss(self, visual_features, semantic_features):
        return mmd(visual_features, semantic_features, **self._gmm_config_loss)

    def _train_gmm(self):
        logger.info("Training GMM.")

        optimizer = factory.get_optimizer(
            [
                {
                    "params": self._network.word_embeddings.parameters(),
                    "lr": self._gmm_config_loss["lr"]
                },
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._gmm_config_loss["convnet_lr"]
                }
            ], self._gmm_config_loss["optimizer"], self._gmm_config_loss["lr"], self._weight_decay
        )

        loops.perclass_loop(
            self.inc_dataset,
            list(range(self._n_classes, self._total_n_classes)),
            self._multiple_devices,
            self._network,
            self._gmm_config_loss["epochs"],
            optimizer,
            self._gmm_loss,
            self._task,
            self._n_tasks,
            target_to_word=self.get_class_label
        )

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        if self._generate_unseen:
            logger.info("Generating weights for unseen classes.")
            real_clf_weights = copy.deepcopy(self._network.classifier._weights)
            self._network.classifier._weights.append(
                nn.Parameter(self.get_fake_weights(**self._generate_unseen))
            )

        for input_dict in loader:
            with torch.no_grad():
                logits = self._network(input_dict["inputs"].to(self._device))["logits"]

            ytrue.append(input_dict["targets"].numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        if self._generate_unseen:
            self._network.classifier._weights = real_clf_weights

        return ypred, ytrue

    def get_fake_weights(self, nb_samples=100, method="word_embeddings", **kwargs):
        classes_to_create = list(range(self._n_classes, self._total_n_classes))

        if method == "word_embeddings":
            weights = []
            for class_id in classes_to_create:
                weights.append(
                    self._network.word_embeddings(
                        (torch.ones(nb_samples,) * class_id).long().to(self._device)
                    ).mean(dim=0, keepdims=True)
                )
            weights = torch.cat(weights, dim=0)
        elif method == "random":
            weights = torch.randn(len(classes_to_create), 128).float().to(self._device)
        else:
            raise NotImplementedError(
                "Unknown method {} to generate unseen weights.".format(method)
            )

        return weights

    def _accuracy(self, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(dim=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _forward_loss(self, training_network, inputs, targets, memory_flags, metrics, **kwargs):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags, metrics)

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, metrics):
        logits = outputs["logits"]
        scaled_logits = self._network.post_process(logits)

        loss = F.cross_entropy(scaled_logits, targets)
        metrics["clf"] += loss.item()

        return loss


def mmd(x, y, sigmas=[1, 5, 10], normalize=True, scale_matrix=False, **kwargs):
    """Maximum Mean Discrepancy with several Gaussian kernels."""
    xy = torch.cat((x, y))

    if normalize:
        xy = F.normalize(xy, p=2, dim=1)
    if scale_matrix:
        scale = get_scale_matrix(x.shape[0], y.shape[0], x.device)
    else:
        scale = 1.

    factors = _get_mmd_sigmas(sigmas, x.device)

    xx = torch.mm(xy, xy.t())
    x2 = torch.sum(xx**2, dim=1, keepdim=True)

    exponent = xx - 0.5 * x2 - 0.5 * x2.t()

    loss = 0.
    for sigma in sigmas:
        kernel_val = torch.exp(1 / sigma * exponent)
        loss += torch.sum(scale * kernel_val)
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

    return torch.sqrt(loss)


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_sigmas(sigmas, device):
    sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
    return -1 / (2 * sigmas)


@functools.lru_cache(maxsize=1, typed=False)
def get_scale_matrix(M, N, device):
    s1 = (torch.ones((N, 1)) * 1.0 / N)
    s2 = (torch.ones((M, 1)) * -1.0 / M)
    return torch.cat((s1, s2), 0).to(device)
