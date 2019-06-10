import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from inclearn import factory, utils
from inclearn.lib import callbacks, network
from inclearn.models.base import IncrementalLearner

tqdm.monitor_interval = 0


class Medic(IncrementalLearner):
    """Implementation of:

    - Incremental Learning with Maximum Entropy Regularization: Rethinking
      Forgetting and Intransigence.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._k = args["memory_size"]
        self._n_classes = 0
        self._epochs = args["epochs"]

        self._network = network.BasicNet(
            args["convnet"], use_bias=True, use_multi_fc=False, device=self._device
        )

        self._examplars = {}
        self._old_model = []

        self._task_idxes = []

    # ----------
    # Public API
    # ----------

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def _before_task(self, train_loader, val_loader):
        """Set up before the task training can begin.

        1. Precomputes previous model probabilities.
        2. Extend the classifier to support new classes.

        :param train_loader: The training dataloader.
        :param val_loader: The validation dataloader.
        """
        self._network.add_classes(self._task_size)

        self._task_idxes.append([self._n_classes + i for i in range(self._task_size)])

        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._m))

    def _train_task(self, train_loader, val_loader):
        """Train & fine-tune model.

        :param train_loader: A DataLoader.
        :param val_loader: A DataLoader, can be None.
        """
        optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self._scheduling, gamma=self._lr_decay
        )
        self._train(train_loader, val_loader, self._epochs, optimizer, scheduler)

    def _after_task(self, data_loader):
        self._reduce_examplars()
        self._build_examplars(data_loader)
        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        return ypred, ytrue

    def get_memory_indexes(self):
        return self.examplars

    # -----------
    # Private API
    # -----------

    def _train(self, train_loader, val_loader, n_epochs, optimizer, scheduler):
        self._best_acc = float("-inf")

        print("nb ", len(train_loader.dataset))

        val_acc = 0.
        train_acc = 0.
        for epoch in range(n_epochs):
            scheduler.step()

            _clf_loss, _distil_loss = 0., 0.
            c = 0

            for i, ((_, idxes), inputs, targets) in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                c += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    inputs,
                    logits,
                    targets,
                    idxes,
                )

                if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                    import pdb
                    pdb.set_trace()

                loss = clf_loss + distil_loss

                loss.backward()

                optimizer.step()

                _clf_loss += clf_loss.item()
                _distil_loss += distil_loss.item()

            if val_loader:
                self._network.eval()
                ypred, ytrue = self._classify(val_loader)
                val_acc = (ypred == ytrue).sum() / len(ytrue)
                self._best_acc = max(self._best_acc, val_acc)
                ypred, ytrue = self._classify(train_loader)
                train_acc = (ypred == ytrue).sum() / len(ytrue)
                self._network.train()

            print("Epoch {}/{}; Clf: {}; Distill: {}; Train: {}; Val: {}".format(
                    epoch, n_epochs,
                    round(_clf_loss / c, 3),
                    round(_distil_loss / c, 3),
                    round(train_acc, 3),
                    round(val_acc, 3),
                )
            )

        print("best", self._best_acc)

    def _compute_loss(self, inputs, logits, targets, idxes):
        """Computes the classification loss & the distillation loss.

        Distillation loss is null at the first task.

        :param logits: Logits produced the model.
        :param targets: The targets.
        :param idxes: The real indexes of the just-processed images. Needed to
                      match the previous predictions.
        :return: A tuple of the classification loss and the distillation loss.
        """
        clf_loss = F.cross_entropy(logits, targets)

        if self._task == 0:
            distil_loss = torch.zeros(1, device=self._device)
        else:
            last_index = len(self._task_idxes) - 1

            distil_loss = 0.
            with torch.no_grad():
                previous_logits = self._old_model(inputs)

            for i in range(last_index):
                task_idxes = self._task_idxes[i]

                ce_loss = F.binary_cross_entropy(
                    F.softmax(logits[..., task_idxes], dim=1),
                    F.softmax(previous_logits[..., task_idxes], dim=1)
                )
                entropy_loss = self.entropy(logits[..., task_idxes])

                mer_loss = ce_loss - entropy_loss
                if mer_loss < 0:
                    import pdb; pdb.set_trace()

                distil_loss += mer_loss

        return clf_loss, distil_loss

    @staticmethod
    def entropy(p):
        e = F.softmax(p, dim=1) * F.log_softmax(p, dim=1)
        return -1.0 * e.mean()

    def _compute_predictions(self, loader):
        """Precomputes the logits before a task.

        :param data_loader: A DataLoader.
        :return: A tensor storing the whole current dataset logits.
        """
        logits = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            logits[idxes] = self._network(inputs).detach()

        return logits

    def _classify(self, loader):
        """Classify the images given by the data loader.

        :param data_loader: A DataLoader.
        :return: A numpy array of the predicted targets and a numpy array of the
                 ground-truth targets.
        """
        ypred = []
        ytrue = []

        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = F.softmax(self._network(inputs), dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _build_examplars(self, loader, n_examplars=None):
        """Builds new examplars.

        :param loader: A DataLoader.
        :param n_examplars: Maximum number of examplars to create.
        """
        n_examplars = n_examplars or self._m

        lo, hi = self._task * self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):
            loader.dataset.set_classes_range(class_idx, class_idx)
            self._examplars[class_idx] = self._build_class_examplars(loader, n_examplars)

    def _build_class_examplars(self, loader, n_examplars):
        """Build examplars for a single class.

        Examplars are selected as the closest to the class mean.

        :param loader: DataLoader that provides images for a single class.
        :param n_examplars: Maximum number of examplars to create.
        :return: The real indexes of the chosen examplars.
        """
        idxes = []
        for (real_idxes, _), _, _ in loader:
            idxes.extend(real_idxes.numpy().tolist())
        idxes = np.array(idxes)

        nb_examplars = min(n_examplars, len(idxes))

        np.random.shuffle(idxes)
        return idxes[:nb_examplars]

    @property
    def examplars(self):
        """Returns all the real examplars indexes.

        :return: A numpy array of indexes.
        """
        return np.array(
            [
                examplar_idx for class_examplars in self._examplars.values()
                for examplar_idx in class_examplars
            ]
        )

    def _reduce_examplars(self):
        print("Reducing examplars.")
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][:self._m]
