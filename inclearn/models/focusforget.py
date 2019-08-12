import collections

import numpy as np
import sklearn
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn import factory, utils
from inclearn.models.base import IncrementalLearner


class FocusForget(IncrementalLearner):
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

        self._temperature = 2.#args["temperature"]

        self._features_extractor = factory.get_resnet(
            args["convnet"], nf=64, zero_init_residual=True
        )
        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes, bias=True)

        self._examplars = {}

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

            self._add_n_classes(self._task_size)

        self._forget_counter = collections.defaultdict(int)
        self._learned_once_flag = collections.defaultdict(bool)
        self._learned_flag = collections.defaultdict(bool)
        self._seen_flag = collections.defaultdict(bool)

    def _train_task(self, train_loader, val_loader):
        # Training on all new + examplars
        self._best_acc = float("-inf")

        print("Training")
        self._finetuning = False
        epochs = 70
        optimizer = factory.get_optimizer(self.parameters(), self._opt_name, 0.1, 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 60], gamma=0.2)
        self._train(train_loader, val_loader, epochs, optimizer, scheduler)

        if self._task == 0:
            print("best", self._best_acc)
            return

        # Fine-tuning on sub-set new + examplars
        print("Fine-tuning")
        self._finetuning = True
        epochs = 50
        self._build_examplars(train_loader,
                              n_examplars=self._k // (self._n_classes - self._task_size))
        train_loader.dataset.set_idxes(self.examplars)  # Fine-tuning only on balanced dataset
        self._previous_preds = self._compute_predictions(train_loader)

        optimizer = factory.get_optimizer(self.parameters(), self._opt_name, 0.01, 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.2)
        self._train(train_loader, val_loader, epochs, optimizer, scheduler)

        print("best", self._best_acc)

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

    def _train(self, train_loader, val_loader, n_epochs, optimizer, scheduler):
        uniq = set()

        print("nb ", len(train_loader.dataset))
        prog_bar = trange(n_epochs, desc="Losses.")

        val_acc = 0.
        for epoch in prog_bar:
            if epoch % 10 == 0 and val_loader:
                ypred, ytrue = self._classify(val_loader)
                val_acc = (ypred == ytrue).sum() / len(ytrue)
                self._best_acc = max(self._best_acc, val_acc)

            _clf_loss, _distil_loss = 0., 0.
            c = 0

            scheduler.step()

            for i, ((real_idxes, idxes), inputs, targets) in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                c += len(idxes)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self.forward(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    logits,
                    targets,
                    idxes
                )

                if not self._finetuning:
                    self._track_forgetting(logits, targets, real_idxes)

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
                        "Clf loss: {}; Distill loss: {}; Val acc: {}".format(
                            round(clf_loss.item(), 3), round(distil_loss.item(), 3),
                            round(val_acc, 3)
                        )
                    )

            prog_bar.set_description(
                "Clf loss: {}; Distill loss: {}; Val acc: {}".format(
                    round(_clf_loss / c, 3), round(_distil_loss / c, 3),
                    round(val_acc, 3)
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
        clf_loss = F.cross_entropy(logits, targets)

        if self._task == 0:
            distil_loss = torch.zeros(1, device=self._device)
        else:
            if not self._finetuning:
                logits = logits[..., :self._new_task_index]

            distil_loss = F.binary_cross_entropy(
                F.softmax(logits / self._temperature, dim=1),
                F.softmax(self._previous_preds[idxes] / self._temperature, dim=1)
            )

        return clf_loss, distil_loss

    def _track_forgetting(self, logits, targets, idxes):
        """real idxes !"""
        targets = targets.cpu().numpy()
        preds = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        idxes = idxes.cpu().numpy()

        are_right = targets == preds

        assert len(targets) == len(preds) == len(idxes)
        for is_right, idx in zip(are_right, idxes):
            self._seen_flag[idx] = True
            if is_right == 1:  # Good prediction
                self._learned_once_flag[idx] = True
                self._learned_flag[idx] = True
            elif is_right == 0:  # Bad prediction
                if idx in self._learned_flag and self._learned_flag[idx]:
                    # Was good at previous epoch
                    self._forget_counter[idx] += 1
            else:
                print("Oops")
                import pdb; pdb.set_trace()

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
            preds = logits.argmax(dim=1).cpu().numpy()

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _add_n_classes(self, n):
        self._n_classes += n

        weights = self._classifier.weight.data.clone()
        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes,
                                     bias=True).to(self._device)
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
        features, class_mean, idxes = self._extract_features(loader)  # FIXME make simpler way to extract indexes

        n_examplars = min(n_examplars, len(idxes))

        """
        n = len(idxes)
        never_learned = []
        for idx in idxes:
            if idx not in self._learned_once_flag:
                if idx not in self._seen_flag:
                    import pdb; pdb.set_trace()
                never_learned.append(idx)

        forget_counter = {}
        for idx in idxes:
            forget_counter[idx] = self._forget_counter[idx]

        forget_counter = sorted(forget_counter.items(), key=lambda x: x[1])
        removed_index = [tup for tup in forget_counter if tup[1] < 10]
        #v = [tup[1] for tup in forget_counter]
        #mean = sum(v) / len(v)
        """

        features = F.normalize(features, dim=1)

        n_clusters = 5

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(features.cpu().numpy())
        clusters = kmeans.cluster_centers_

        selected_idxes = set()
        for cluster_idx in range(n_clusters):
            cluster = torch.tensor(clusters[cluster_idx], device=self._device)
            distances_to_cluster = self._dist(cluster, features)
            fake_idxes = distances_to_cluster.argsort().cpu().numpy()
            fake_idxes = fake_idxes[:n_examplars // n_clusters]

            for idx in fake_idxes:
                selected_idxes.add(idxes[idx])
            # See duplicates

        selected_idxes = list(selected_idxes)
        import random
        random.shuffle(selected_idxes)

        if len(selected_idxes) > n_examplars:
            import pdb; pdb.set_trace()

        return selected_idxes

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

    @staticmethod
    def _dist(a, b):
        """Computes L2 distance between two tensors.

        :param a: A tensor.
        :param b: A tensor.
        :return: A tensor of distance being of the shape of the "biggest" input
                 tensor.
        """
        return torch.pow(a - b, 2).sum(-1)
