import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn.models.base import IncrementalLearner
from inclearn import factory
from inclearn import utils


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    :param args: An argparse parsed arguments object.
    """
    def __init__(self, args):
        super().__init__()

        self._device = args.device
        self._memory_size = args.memory_size
        self._opt_name = args.optimizer
        self._lr = args.lr
        self._n_epochs = args.epochs

        self._k = args.memory_size
        self._n_classes = args.increment

        self._features_extractor = factory.get_resnet(args.convnet)
        self._classifier = nn.Linear(2048, self._n_classes, bias=False)

        self._examplars = {}
        self._means = None

        self._clf_loss = F.binary_cross_entropy_with_logits#nn.CrossEntropyLoss()
        self._distil_loss = F.binary_cross_entropy_with_logits#nn.BCEWithLogitsLoss()

        self.to(self._device)

    def forward(self, x):
        x = self._features_extractor(x)
        x = self._classifier(x)
        return x

    # ----------
    # Public API
    # ----------

    def _before_task(self, data_loader):
        if self._task == 0:
            self._previous_preds = None
        else:
            print("Computing previous predictions...")
            self._previous_preds = self._compute_predictions(data_loader)
            self._add_n_classes(self._task_size)

        self._optimizer = factory.get_optimizer(self.parameters(), self._opt_name, self._lr)

    def _train_task(self, data_loader):
        prog_bar = trange(self._n_epochs, desc="Losses.")

        for epoch in prog_bar:
            _clf_loss, _distil_loss = 0., 0.

            for idx, (idxes, inputs, targets) in enumerate(data_loader, start=1):
                self._optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = utils.to_onehot(targets, self._n_classes).to(self._device)
                logits = self.forward(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    logits,
                    targets,
                    idxes[1],
                )

            if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                import pdb; pdb.set_trace()

            loss = clf_loss + distil_loss

            loss.backward()
            self._optimizer.step()

            _clf_loss += clf_loss.item()
            _distil_loss += distil_loss.item()

            if idx % 10 == 0 or idx >= len(data_loader):
                prog_bar.set_description(
                "Clf loss: {}; Distill loss: {}".format(
                    round(clf_loss.item(), 3),
                    round(distil_loss.item(), 3),
                    )
                )

    def _after_task(self, data_loader):
        self._build_examplars(data_loader)

    def _eval_task(self, data_loader):
        return self._classify(data_loader)

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, logits, targets, idxes):
        if self._task == 0:
            # First task, only doing classification loss
            clf_loss = self._clf_loss(logits, targets)
            distil_loss = torch.zeros(1, device=self._device)
        else:
            try:
                clf_loss = self._clf_loss(
                    logits[..., self._new_task_index:],
                    targets[..., self._new_task_index:]
                )
            except:
                import pdb; pdb.set_trace()

            distil_loss = self._distil_loss(
                logits[..., :self._new_task_index],
                self._previous_preds[idxes, :self._new_task_index]
            )

        return clf_loss, distil_loss


    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self.forward(inputs).detach()

        return torch.sigmoid(preds)


    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError("Cannot classify without built examplar means,"
                             " Have you forgotten to call `before_task`?")
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0])+\
                " with the number of classes ({}).".format(self._n_classes))

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._features_extractor(inputs).detach()
            preds = self._get_closest(self._means, features)

            ypred.extend(preds)
            ytrue.extend(targets)

        return ypred, ytrue

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _add_n_classes(self, n):
        print("add n classes")
        self._n_classes += n

        weight = self._classifier.weight.data
        #bias = self._classifier.bias.data

        self._classifier = nn.Linear(
            self._features_extractor.out_dim, self._n_classes,
            bias=False
        ).to(self._device)

        self._classifier.weight.data[: self._n_classes - n] = weight
        #self._classifier.bias.data[: self._n_classes - n] = bias

        print("Now {} examplars per class.".format(self._m))

    def _extract_features(self, loader):
        features = []

        for _, inputs, _ in loader:
            inputs = inputs.to(self._device)
            features.append(self._features_extractor(inputs).detach())

        features = torch.cat(features)
        mean = torch.mean(features, dim=0, keepdim=True)

        return F.normalize(features), F.normalize(mean)[0]

    @staticmethod
    def _remove_row(matrix, row_idx):
        tmp = torch.cat((matrix[:row_idx, ...], matrix[row_idx + 1 :, ...]))
        del matrix
        return tmp

    @staticmethod
    def _get_closest(centers, features):
        pred_labels = []

        for feature in features:
            distances = torch.pow(centers - feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _get_closest_features(center, features):
        normalized_features = F.normalize(features)
        distances = torch.pow(center - normalized_features, 2).sum(-1)
        return distances.argmin().item()

    def _build_examplars(self, loader):
        examplars = []
        examplars_means = []

        self.eval()
        for class_idx in range(0, self._n_classes):
            loader.dataset.set_classes_range(class_idx, class_idx)

            features, class_mean = self._extract_features(loader)
            examplars_mean = torch.zeros(self._features_extractor.out_dim, device=self._device)

            for i in range(min(self._m, features.shape[0])):
                idx = self._get_closest_features(
                    class_mean, (features + examplars_mean) / (i + 1)
                )
                examplars.append(loader.dataset.get_true_index(idx))
                examplars_mean += features[idx]
                features = self._remove_row(features, idx)

            examplars_means.append(examplars_mean / len(examplars))
            self._examplars[class_idx] = examplars

        self._means = torch.stack(examplars_means)
        self._means = F.normalize(self._means)

        self.train()

    @property
    def examplars(self):
        return np.array(
            [
                examplar_idx
                for class_examplars in self._examplars.values()
                for examplar_idx in class_examplars
            ]
        )

    def reduce_examplars(self):
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][: self._m]
