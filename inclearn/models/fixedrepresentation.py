import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn import factory, utils
from inclearn.models.base import IncrementalLearner

LOGGER = logging.Logger("IncLearn", level="INFO")


class FixedRepresentation(IncrementalLearner):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, args):
        super().__init__()

        self._epochs = 70

        self._n_classes = args["increment"]
        self._device = args["device"]

        self._features_extractor = factory.get_resnet(args["convnet"], nf=64,
                                                      zero_init_residual=True)

        self._classifiers = [nn.Linear(self._features_extractor.out_dim, self._n_classes, bias=False).to(self._device)]
        torch.nn.init.kaiming_normal_(self._classifiers[0].weight)
        self.add_module("clf_" + str(self._n_classes), self._classifiers[0])

        self.to(self._device)

    def forward(self, x):
        feats = self._features_extractor(x)

        logits = []
        for clf in self._classifiers:
            logits.append(clf(feats))

        return torch.cat(logits, dim=1)

    def _before_task(self, data_loader, val_loader):
        if self._task != 0:
            self._add_n_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            filter(lambda x: x.requires_grad, self.parameters()),
            "sgd", 0.1)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [50, 60], gamma=0.2)

    def _get_params(self):
        return [self._features_extractor.parameters()]

    def _train_task(self, train_loader, val_loader):
        for _ in trange(self._epochs):
            self._scheduler.step()
            for _, inputs, targets in train_loader:
                self._optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self.forward(inputs)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                self._optimizer.step()

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, loader):
        ypred = []
        ytrue = []

        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = self.forward(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()

            ypred.extend(preds)
            ytrue.extend(targets)

        ypred, ytrue = np.array(ypred), np.array(ytrue)
        print(np.bincount(ypred))
        return ypred, ytrue

    def _add_n_classes(self, n):
        self._n_classes += n

        self._classifiers.append(nn.Linear(
            self._features_extractor.out_dim, self._task_size,
            bias=False
        ).to(self._device))
        nn.init.kaiming_normal_(self._classifiers[-1].weight)
        self.add_module("clf_" + str(self._n_classes), self._classifiers[-1])

        for param in self._features_extractor.parameters():
            param.requires_grad = False

        for clf in self._classifiers[:-1]:
            for param in clf.parameters():
                param.requires_grad = False
        for param in self._classifiers[-1].parameters():
            for param in clf.parameters():
                param.requires_grad = True
