import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn import factory, utils
from inclearn.models.base import IncrementalLearner


class End2End(IncrementalLearner):
    """Implementation of End-to-End Increment Learning.

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
        self._n_classes = args["increment"]

        self._temperature = args["temperature"]

        self._features_extractor = factory.get_resnet(
            args["convnet"], nf=64, zero_init_residual=True
        )
        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes, bias=False)
        torch.nn.init.kaiming_normal_(self._classifier.weight)

        self._examplars = {}
        self._means = None

        self.to(self._device)

    def forward(self, x):
        x = self._features_extractor(x)
        x = self._classifier(x)
        return x

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        """Set up before the task training can begin.

        1. Precomputes previous model probabilities.
        2. Extend the classifier to support new classes.

        :param train_loader: The training dataloader.
        :param val_loader: The validation dataloader.
        """
        if self._task == 0:
            self._previous_preds = None
        else:
            print("Computing previous predictions...")
            self._previous_preds = self._compute_predictions(train_loader)
            if val_loader:
                self._previous_preds_val = self._compute_predictions(val_loader)

            self._add_n_classes(self._task_size)

    def _train_task(self, train_loader, val_loader):
        # Training on all new + examplars
        self.foo = 0
        optimizer = factory.get_optimizer(self.parameters(), self._opt_name, 0.1, 0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1)
        self._train(train_loader, 1, optimizer, scheduler)

        if self._task == 0:
            return

        # Fine-tuning on sub-set new + examplars
        self._build_examplars(train_loader)
        train_loader.dataset.set_idxes(self.examplars)  # Fine-tuning only on balanced dataset
        optimizer = factory.get_optimizer(self.parameters(), self._opt_name, 0.01, 0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
        self.foo = 1
        self._train(train_loader, 1, optimizer, scheduler)

    def _after_task(self, data_loader):
        self._reduce_examplars()
        self._build_examplars(data_loader)

    def _eval_task(self, data_loader):
        ypred, ytrue = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        return ypred, ytrue

    def get_memory_indexes(self):
        return self.examplars

    # -----------
    # Private API
    # -----------

    def _train(self, train_loader, n_epochs, optimizer, scheduler):
        print("nb ", len(train_loader.dataset))

        prog_bar = trange(n_epochs, desc="Losses.")

        for epoch in prog_bar:
            _clf_loss, _distil_loss = 0., 0.
            c = 0

            scheduler.step()

            for i, ((_, idxes), inputs, targets) in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                c += len(idxes)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self.forward(inputs)

                clf_loss, distil_loss = self._compute_loss(
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

                if i % 10 == 0 or i >= len(train_loader):
                    prog_bar.set_description(
                        "Clf loss: {}; Distill loss: {}".format(
                            round(clf_loss.item(), 3), round(distil_loss.item(), 3)
                        )
                    )

            prog_bar.set_description(
                "Clf loss: {}; Distill loss: {}".format(
                    round(_clf_loss / c, 3), round(_distil_loss / c, 3)
                )
            )

    def _compute_loss(self, logits, targets, idxes):
        """Computes the classification loss & the distillation loss.

        Distillation loss is null at the first task.

        :param logits: Logits produced the model.
        :param targets: The targets.
        :param idxes: The real indexes of the just-processed images. Needed to
                      match the previous predictions.
        :return: A tuple of the classification loss and the distillation loss.
        """
        if self._task == 0:
            clf_loss = F.cross_entropy(logits, targets)
            distil_loss = torch.zeros(1, device=self._device)
        else:
            # Disable the cross_entropy loss for the old targets:
            for i in range(self._new_task_index):
                targets[targets == i] = -1
            clf_loss = F.cross_entropy(logits, targets, ignore_index=-1)

            distil_loss = F.binary_cross_entropy(
                F.softmax(logits[..., :self._new_task_index] ** (1 / self._temperature), dim=1),
                F.softmax(self._previous_preds[idxes]**(1 / self._temperature), dim=1)
            )

        return clf_loss, distil_loss

    def _compute_predictions(self, loader):
        """Precomputes the logits before a task.

        :param data_loader: A DataLoader.
        :return: A tensor storing the whole current dataset logits.
        """
        logits = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            logits[idxes] = self.forward(inputs).detach()

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
            logits = self.forward(inputs)
            preds = F.softmax(logits, dim=1).argmax(dim=1)

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _add_n_classes(self, n):
        self._n_classes += n

        weights = self._classifier.weight.data
        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes,
                                     bias=False).to(self._device)
        torch.nn.init.kaiming_normal_(self._classifier.weight)

        self._classifier.weight.data[:self._n_classes - n] = weights

        print("Now {} examplars per class.".format(self._m))

    def _extract_features(self, loader):
        features = []
        idxes = []

        for (real_idxes, _), inputs, _ in loader:
            inputs = inputs.to(self._device)
            features.append(self._features_extractor(inputs).detach())
            idxes.extend(real_idxes.numpy().tolist())

        features = torch.cat(features)
        mean = torch.mean(features, dim=0, keepdim=False)

        return features, mean, idxes

    @staticmethod
    def _get_closest(centers, features):
        """Returns the center index being the closest to each feature.

        :param centers: Centers to compare, in this case the class means.
        :param features: A tensor of features extracted by the convnet.
        :return: A numpy array of the closest centers indexes.
        """
        pred_labels = []

        features = features
        for feature in features:
            distances = End2End._dist(centers, feature)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _dist(a, b):
        """Computes L2 distance between two tensors.

        :param a: A tensor.
        :param b: A tensor.
        :return: A tensor of distance being of the shape of the "biggest" input
                 tensor.
        """
        return torch.pow(a - b, 2).sum(-1)

    def _build_examplars(self, loader):
        """Builds new examplars.

        :param loader: A DataLoader.
        """
        lo, hi = self._task * self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):
            loader.dataset.set_classes_range(class_idx, class_idx)
            self._examplars[class_idx] = self._build_class_examplars(loader)

    def _build_class_examplars(self, loader):
        """Build examplars for a single class.

        Examplars are selected as the closest to the class mean.

        :param loader: DataLoader that provides images for a single class.
        :return: The real indexes of the chosen examplars.
        """
        features, class_mean, idxes = self._extract_features(loader)

        class_mean = F.normalize(class_mean, dim=0)
        distances_to_mean = self._dist(class_mean, features)

        nb_examplars = min(self._m, len(features))

        fake_idxes = distances_to_mean.argsort().cpu().numpy()[:nb_examplars]
        return [idxes[idx] for idx in fake_idxes]

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
