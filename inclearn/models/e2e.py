import numpy as np
import torch
import tqdm
from torch.nn import functional as F

from inclearn.lib import factory, herding, network, utils
from inclearn.models.base import IncrementalLearner

tqdm.monitor_interval = 0


class End2End(IncrementalLearner):
    """Implementation of End-to-End Increment Learning.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        #self._lr = args["lr"]
        #self._weight_decay = args["weight_decay"]
        #self._n_epochs = args["epochs"]

        #self._scheduling = args["scheduling"]
        #self._lr_decay = args["lr_decay"]

        self._k = args["memory_size"]
        self._n_classes = 0

        self._temperature = args["temperature"]

        self._network = network.BasicNet(
            args["convnet"],
            use_bias=True,
            use_multi_fc=True,
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False
        )

        self._data_memory, self._targets_memory = {}, {}
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
        print("Now {} examplars per class.".format(self._memory_per_class))

    def _train_task(self, train_loader, val_loader):
        """Train & fine-tune model.

        The scheduling is different from the paper for one reason. In the paper,
        End-to-End Incremental Learning, the authors pre-generated 12 augmentations
        per images (thus multiplying by this number the dataset size). However
        I find this inefficient for large scale datasets, thus I'm simply doing
        the augmentations online. A greater number of epochs is then needed to
        match performances.

        :param train_loader: A DataLoader.
        :param val_loader: A DataLoader, can be None.
        """
        if self._task == 0:
            epochs = 90
            optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.1, 0.0001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 60], gamma=0.1)
            self._train(train_loader, val_loader, epochs, optimizer, scheduler)
            return

        # Training on all new + examplars
        print("Training")
        self._finetuning = False
        epochs = 60
        optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.1, 0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 50], gamma=0.1)
        self._train(train_loader, val_loader, epochs, optimizer, scheduler)

        # Fine-tuning on sub-set new + examplars
        print("Fine-tuning")
        self._old_model = self._network.copy().freeze()

        self._finetuning = True
        self._build_examplars(n_examplars=self._k // (self._n_classes - self._task_size))

        loader = self.inc_dataset.get_memory_loader(*self.get_memory())
        optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.01, 0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
        self._train(loader, val_loader, 40, optimizer, scheduler)

    def _after_task(self, inc_dataset):
        self._reduce_examplars()
        self._build_examplars()

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        return ypred, ytrue

    def get_memory(self):
        data = np.concatenate(list(self._data_memory.values()))
        targets = np.concatenate(list(self._targets_memory.values()))

        return data, targets

    # -----------
    # Private API
    # -----------

    def _train(self, train_loader, val_loader, n_epochs, optimizer, scheduler):
        for p in self._network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -2., 2.))

        self._callbacks = [
            #callbacks.GaussianNoiseAnnealing(self._network.parameters()),
            #callbacks.EarlyStopping(self._network, minimize_metric=False)
        ]
        self._best_acc = float("-inf")

        print("nb ", len(train_loader.dataset))

        val_acc = 0.
        train_acc = 0.
        for epoch in range(n_epochs):
            for cb in self._callbacks:
                cb.on_epoch_begin()

            scheduler.step()

            _clf_loss, _distil_loss = 0., 0.
            c = 0

            prog_bar = tqdm.tqdm(enumerate(train_loader, start=1), desc="Losses.")

            for i, (inputs, targets, _) in prog_bar:
                optimizer.zero_grad()

                c += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    inputs,
                    logits,
                    targets
                )

                if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                    import pdb
                    pdb.set_trace()

                loss = clf_loss + distil_loss

                loss.backward(retain_graph=True)

                #l2reg_grad = 0.
                #for p in self._network.parameters():
                #    l2reg_grad += 0.0001 * p.grad.norm()
                #loss += l2reg_grad
                #loss.backward()

                for cb in self._callbacks:
                    cb.before_step()
                optimizer.step()

                _clf_loss += clf_loss.item()
                _distil_loss += distil_loss.item()

                if i % 10 == 0 or i >= len(train_loader):
                    prog_bar.set_description(
                        "Task {}/{}; Epoch {}/{}: Clf: {}; Distill: {}; Train: {}; Val: {}".format(
                            self._task + 1, self._n_tasks,
                            epoch + 1, n_epochs,
                            round(clf_loss.item(), 3), round(distil_loss.item(), 3),
                            round(train_acc, 3),
                            round(val_acc, 3)
                        )
                    )

            if val_loader:
                ypred, ytrue = self._classify(val_loader)
                val_acc = (ypred == ytrue).sum() / len(ytrue)
                self._best_acc = max(self._best_acc, val_acc)
                ypred, ytrue = self._classify(train_loader)
                train_acc = (ypred == ytrue).sum() / len(ytrue)

            for cb in self._callbacks:
                cb.on_epoch_end(metric=val_acc)

            prog_bar.set_description(
                "Clf: {}; Distill: {}; Train: {}; Val: {}".format(
                    round(_clf_loss / c, 3), round(_distil_loss / c, 3),
                    round(train_acc, 3),
                    round(val_acc, 3),
                )
            )

            for cb in self._callbacks:
                if not cb.in_training:
                    self._network = cb.network
                    return

    def _compute_loss(self, inputs, logits, targets):
        """Computes the classification loss & the distillation loss.

        Distillation loss is null at the first task.

        :param logits: Logits produced the model.
        :param targets: The targets.
        :return: A tuple of the classification loss and the distillation loss.
        """
        clf_loss = F.cross_entropy(logits, targets)

        if clf_loss.item() > 100:
            import pdb; pdb.set_trace()

        #return clf_loss, torch.zeros(1, device=self._device)
        if self._task == 0:
            distil_loss = torch.zeros(1, device=self._device)
        else:
            if self._finetuning:
                # We only do distillation on current task during the distillation
                # phase:
                last_index = len(self._task_idxes)
                n = self._n_classes
            else:
                last_index = len(self._task_idxes) - 1
                n = self._n_classes - self._task_size

            distil_loss = 0.
            with torch.no_grad():
                previous_logits = self._old_model(inputs)

            for i in range(last_index):
                task_idxes = self._task_idxes[i]

                task_prob_new = F.softmax(logits[..., task_idxes], dim=1)
                task_prob_old = F.softmax(previous_logits[..., task_idxes], dim=1)

                task_prob_new = task_prob_new ** (1 / self._temperature)
                task_prob_old = task_prob_old ** (1 / self._temperature)

                task_prob_new = task_prob_new / task_prob_new.sum(1).view(-1, 1)
                task_prob_old = task_prob_old / task_prob_old.sum(1).view(-1, 1)

                distil_loss += F.binary_cross_entropy(
                    task_prob_new,
                    task_prob_old
                )

            distil_loss *= 1 / last_index

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

        for inputs, targets, _ in loader:
            inputs = inputs.to(self._device)
            logits = F.softmax(self._network(inputs), dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _build_examplars(self, n_examplars=None):
        """Builds new examplars.

        :param loader: An incremental dataset.
        :param n_examplars: Maximum number of examplars to create.
        :return: The memory data and the associated targets.
        """
        n_examplars = n_examplars or self._memory_per_class

        lo, hi = self._n_classes - self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):
            inputs, loader = self.inc_dataset.get_custom_loader(class_idx, mode="test")
            features, targets = extract_features(self._network, loader)

            indexes = herding.closest_to_mean(features, n_examplars)

            self._data_memory[class_idx] = inputs[indexes]
            self._targets_memory[class_idx] = targets[indexes]

    def _reduce_examplars(self):
        print("Reducing examplars.")
        for class_idx in self._data_memory.keys():
            self._data_memory[class_idx] = self._data_memory[class_idx][:self._memory_per_class]
            self._targets_memory[class_idx] = self._targets_memory[class_idx][:self._memory_per_class]


def extract_features(model, loader):
    targets, features = [], []

    for _inputs, _targets, _ in loader:
        _targets = _targets.numpy()
        _features = model.extract(_inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)
