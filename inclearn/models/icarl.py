import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn import factory, utils
from inclearn.lib import network
from inclearn.models.base import IncrementalLearner


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

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._k = args["memory_size"]
        self._n_classes = 0

        self._network = network.BasicNet(args["convnet"], device=self._device)

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        print("Now {} examplars per class.".format(self._m))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, train_loader, val_loader):
        print("nb ", len(train_loader.dataset))

        prog_bar = trange(self._n_epochs, desc="Losses.")

        for epoch in prog_bar:
            _loss = 0.
            val_loss = 0.

            self._scheduler.step()

            for inputs, targets in train_loader:
                self._optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = utils.to_onehot(targets, self._n_classes).to(self._device)
                logits = self._network(inputs)

                loss = self._compute_loss(
                    inputs,
                    logits,
                    targets
                )

                if not utils._check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                _loss += loss.item()

            prog_bar.set_description(
                "Clf loss: {}; Val loss: {}".format(
                    round(_loss / len(train_loader), 3), round(val_loss, 2)
                )
            )

    def _after_task(self, data_loader):
        self._reduce_examplars()
        self._build_examplars(data_loader)

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        return ypred, ytrue

    def get_memory(self):
        return self.examplars

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets):
        targets = utils.one_hot(targets, self._n_classes)

        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())

            new_targets = targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

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
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _extract_features(self, loader):
        features = []
        idxes = []

        for (real_idxes, _), inputs, _ in loader:
            inputs = inputs.to(self._device)
            features.append(self._network.extract(inputs).detach())
            idxes.extend(real_idxes.numpy().tolist())

        features = F.normalize(torch.cat(features), dim=1)
        mean = torch.mean(features, dim=0, keepdim=False)

        return features, mean, idxes

    @staticmethod
    def _get_closest(centers, features):
        pred_labels = []

        features = features
        for feature in features:
            distances = ICarl._dist(feature, centers)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _dist(one, many):
        return -(one @ many.transpose(1, 0)).squeeze()
        #return torch.pow(one - many, 2).sum(-1)

    def _build_examplars(self, loader):
        #loader.dataset._use_data_augmentation = False

        means = []


        lo, hi = 0, self._task * self._task_size
        print("Updating examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):

            loader.dataset.set_idxes(self._examplars[class_idx])
            # loader.dataset._flip_all = True
            #loader.dataset.double_dataset()
            _, examplar_mean, _ = self._extract_features(loader)
            means.append(F.normalize(examplar_mean, dim=0))
            #loader.dataset._flip_all = False

        lo, hi = self._task * self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):
            examplars_idxes = []

            loader.dataset.set_classes_range(class_idx, class_idx)

            features, class_mean, idxes = self._extract_features(loader)
            examplars_mean = torch.zeros(self._network.convnet.out_dim, device=self._device)

            class_mean = F.normalize(class_mean, dim=0)
            """
            # Icarl
            for i in range(min(self._m, features.shape[0])):
                tmp = F.normalize(
                    (features + examplars_mean) / (i + 1),
                    dim=1
                )
                distances = self._dist(class_mean, tmp)
                idxes_winner = distances.argsort().cpu().numpy()

                for idx in idxes_winner:
                    real_idx = idxes[idx]
                    if real_idx in examplars_idxes:
                        continue

                    examplars_idxes.append(real_idx)
                    examplars_mean += features[idx]
                    break
            """

            # icarl rebuffi
            n_iter = 0
            w_t = class_mean.clone()
            origi = []
            while (len(examplars_idxes) < min(self._m, features.shape[0]) and n_iter < 1000):
                tmp_t = (w_t @ features.transpose(1, 0)).squeeze()
                idx_max = tmp_t.argmax()

                n_iter += 1

                if idxes[idx_max] not in examplars_idxes:
                    examplars_idxes.append(idxes[idx_max])
                    examplars_mean += features[idx_max]
                    origi.append(idx_max)

                w_t = w_t + class_mean - features[idx_max]

            if len(examplars_idxes) < min(self._m, features.shape[0]):
                remaining = [idx for idx in list(range(features.shape[0])) if idx not in origi]
                import random
                random.shuffle(remaining)
                missing = min(self._m, features.shape[0]) - len(examplars_idxes)
                for i in range(missing):
                    examplars_mean += features[remaining[i]]
                    examplars_idxes.append(idxes[remaining[i]])

            #loader.dataset.set_idxes(examplars_idxes)
            #loader.dataset.double_dataset()
            #loader.dataset._flip_all = True
            #features, class_mean, idxes = self._extract_features(loader)
            #means.append(F.normalize(class_mean, dim=0))
            #loader.dataset._flip_all = False
            """
            # random
            fake_idxes = [i for i in range(features.shape[0])]
            import random
            random.shuffle(fake_idxes)
            fake_idxes = fake_idxes[:min(self._m, features.shape[0])]
            examplars_idxes = [idxes[idx] for idx in fake_idxes]
            for idx in fake_idxes:
                examplars_mean += features[idx]
            """
            """
            dists = (class_mean @ features.transpose(1, 0)).squeeze()
            sorted_idxes = dists.argsort(descending=True)[:min(self._m, features.shape[0])]
            for idx in sorted_idxes:
                examplars_idxes.append(idxes[idx])
                examplars_mean += features[idx]
            """

            means.append(F.normalize(examplars_mean / len(examplars_idxes), dim=0))
            self._examplars[class_idx] = examplars_idxes

        self._means = torch.stack(means)
        #loader.dataset._use_data_augmentation = True

    @property
    def examplars(self):
        return np.array(
            [
                examplar_idx for class_examplars in self._examplars.values()
                for examplar_idx in class_examplars
            ]
        )

    def _reduce_examplars(self):
        return
        print("Reducing examplars.")
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][:self._m]




def extract_features(model, dataset):
    gen = DataLoader(dataset, shuffle=False, batch_size=256)
    features = []
    targets_all = []
    for inputs, targets in gen:
        features.append(model.extract(inputs.to(model.device)).detach())
        targets_all.append(targets.numpy())

    return torch.cat(features), np.concatenate(targets_all)


def extract(model, x, y):
    feat_normal, _ = extract_features(model, IncDataset(x, y, train=None))
    feat_flip, _ = extract_features(model, IncDataset(x, y, train="flip"))

    return feat_normal, feat_flip


def select_examplars(features, nb_max):
    D = features.cpu().numpy().T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_mat = np.zeros((features.shape[0]))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not(np.sum(herding_mat!=0)==min(nb_max,features.shape[0])) and iter_herding_eff<1000:
        tmp_t   = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_mat[ind_max] == 0:
            herding_mat[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    return herding_mat


def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
    D = feat_norm.cpu().numpy().T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.cpu().numpy().T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    mean /= np.linalg.norm(mean)

    return mean, alph


def compute_accuracy(model, test_dataset, class_means):
    features, targets_ = extract_features(model, test_dataset)
    features = features.cpu().numpy()

    targets = np.zeros((targets_.shape[0], 100),np.float32)
    targets[range(len(targets_)),targets_.astype('int32')] = 1.
    features  = (features.T / (np.linalg.norm(features.T,axis=0) + EPSILON)).T

    # Compute score for iCaRL
    sqd         = cdist(class_means, features, 'sqeuclidean')
    score_icarl = (-sqd).T

    # Compute the accuracy over the batch
    stat_icarl = [ll in best for ll, best in zip(targets_.astype('int32'), np.argsort(score_icarl, axis=1)[:, -1:])]

    return np.average(stat_icarl)
