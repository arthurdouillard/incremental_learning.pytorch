import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inclearn.lib import utils


class IncrementalDataset:
    """Incremental generator of datasets.

    :param dataset_name: Among a list of available dataset, that can easily
                         be defined (see at file's end).
    :param random_order: Shuffle the class ordering, else use a cherry-picked
                         ordering.
    :param shuffle: Shuffle batch order between epochs.
    :param workers: Number of workers loading the data.
    :param batch_size: The batch size.
    :param seed: Seed to force determinist class ordering.
    :param increment: Number of class to add at each task.
    :param validation_split: Percent of training data to allocate for validation.
    :param onehot: Returns targets encoded as onehot vectors instead of scalars.
                   Memory is expected to be already given in an onehot format.
    :param initial_increment: Initial increment may be defined if you want to train
                              on more classes than usual for the first task, like
                              UCIR does.
    """

    def __init__(
        self,
        dataset_name,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        onehot=False,
        initial_increment=None
    ):
        datasets = _get_datasets(dataset_name)
        self._setup_data(
            datasets,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment
        )
        self.train_transforms = datasets[0].train_transforms  # FIXME handle multiple datasets
        self.common_transforms = datasets[0].common_transforms

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._onehot = onehot

    @property
    def n_tasks(self):
        return len(self.increments)

    def new_task(self, memory=None, memory_val=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        x_train, y_train = self._select(
            self.data_train, self.targets_train, low_range=min_class, high_range=max_class
        )
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if self._onehot:

            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train = to_onehot(y_train)

        if memory is not None:
            print("Set memory of size: {}.".format(memory[0].shape[0]))
            x_train, y_train, train_memory_flags = self._add_memory(x_train, y_train, *memory)
        else:
            train_memory_flags = np.zeros((x_train.shape[0],))
        if memory_val is not None:
            print("Set validation memory of size: {}.".format(memory_val[0].shape[0]))
            x_val, y_val, val_memory_flags = self._add_memory(x_val, y_val, *memory_val)
        else:
            val_memory_flags = np.zeros((x_val.shape[0],))

        train_loader = self._get_loader(x_train, y_train, train_memory_flags, mode="train")
        val_loader = self._get_loader(x_val, y_val, val_memory_flags,
                                      mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, np.zeros((x_test.shape[0],)), mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _add_memory(self, x, y, data_memory, targets_memory):
        if self._onehot:  # Need to add dummy zeros to match the number of targets:
            targets_memory = np.concatenate(
                (
                    targets_memory,
                    np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )

        memory_flags = np.concatenate((np.zeros((x.shape[0],)), np.ones((data_memory.shape[0],))))

        x = np.concatenate((x, data_memory))
        y = np.concatenate((y, targets_memory))

        return x, y, memory_flags

    def get_custom_loader(self, class_indexes, memory=None, mode="test", data_source="train"):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if memory is not None:
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(data, targets, memory_flags, shuffle=False, mode=mode)

    def get_memory_loader(self, data, targets):
        return self._get_loader(
            data, targets, np.ones((data.shape[0],)), shuffle=True, mode="train"
        )

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _get_loader(self, x, y, memory_flags, shuffle=True, mode="train"):
        if mode == "train":
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
        elif mode == "test":
            trsf = transforms.Compose(self.common_transforms)
        elif mode == "flip":
            trsf = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=1.), *self.common_transforms]
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        return DataLoader(
            DummyDataset(x, y, memory_flags, trsf),
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._workers,
        )

    def _setup_data(
        self,
        datasets,
        random_order=False,
        seed=1,
        increment=10,
        validation_split=0.,
        initial_increment=None
    ):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            train_dataset = dataset.base_dataset("data", train=True, download=True)
            test_dataset = dataset.base_dataset("data", train=False, download=True)

            x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
            x_val, y_val, x_train, y_train = self._split_per_class(
                x_train, y_train, validation_split
            )
            x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

            order = list(range(len(np.unique(y_train))))
            if random_order:
                random.seed(seed)  # Ensure that following order is determined by seed:
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order

            print("Dataset {}: class ordering: {}.".format(dataset, order))

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            elif initial_increment is None:
                self.increments = [increment for _ in range(len(order) // increment)]
            else:
                self.increments = [initial_increment] + [
                    increment
                    for _ in range((len(order) // increment) - (initial_increment // increment))
                ]

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.

        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, memory_flags, trsf):
        self.x, self.y = x, y
        self.memory_flags = memory_flags
        self.trsf = trsf

        assert x.shape[0] == y.shape[0] == memory_flags.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        memory_flag = self.memory_flags[idx]

        x = Image.fromarray(x.astype("uint8"))
        x = self.trsf(x)

        return x, y, memory_flag


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(iCIFAR10):
    base_dataset = datasets.cifar.CIFAR100
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    class_order = [  # Taken from original iCaRL implementation:
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]


class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
    common_transforms = [transforms.ToTensor()]


class iPermutedMNIST(iMNIST):

    def _preprocess_initial_data(self, data):
        b, w, h, c = data.shape
        data = data.reshape(b, -1, c)

        permutation = np.random.permutation(w * h)

        data = data[:, permutation, :]

        return data.reshape(b, w, h, c)


# --------------
# Data utilities
# --------------
