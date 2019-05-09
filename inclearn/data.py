import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# --------
# Datasets
# --------

class IncrementalDataset(torch.utils.data.Dataset):
    _base_dataset = None
    _train_transforms = []
    _common_transforms = [transforms.ToTensor()]

    def __init__(self, data_path="data", train=True, randomize_class=False,
                 increment=10, shuffle=True, workers=10, batch_size=128,
                 classes_order=None):
        self._train = train
        self._increment = increment

        dataset = self._base_dataset(
            data_path,
            train=train,
            download=True,
        )
        self._data = self._preprocess_initial_data(dataset.data)
        self._targets = np.array(dataset.targets)

        if classes_order is None:
            self.classes_order = np.sort(np.unique(self._targets))

            if randomize_class:
                np.random.shuffle(self.classes_order)
        else:
            self.classes_order = classes_order

        trsf = self._train_transforms if train else []
        trsf = trsf + self._common_transforms
        self._transforms = transforms.Compose(trsf)

        self._shuffle = shuffle
        self._workers = workers
        self._batch_size = batch_size

        self._memory_idxes = []

        print("Classes order: ", self.classes_order)
        self.set_classes_range(0, self._increment)

    def get_loader(self, validation_split=0.):
        if validation_split:
            indices = np.arange(len(self))
            np.random.shuffle(indices)
            split_idx = int(len(self) * validation_split)
            val_indices = indices[:split_idx]
            train_indices = indices[split_idx:]
            print("Val {}; Train {}.".format(val_indices.shape[0], train_indices.shape[0]))

            train_loader = self._get_loader(SubsetRandomSampler(train_indices))
            val_loader = self._get_loader(SubsetRandomSampler(val_indices))
            return train_loader, val_loader

        return self._get_loader(), None

    def _get_loader(self, sampler=None):
        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=False if sampler else self._shuffle,
            num_workers=self._workers,
            sampler=sampler
        )

    @property
    def total_n_classes(self):
        return len(np.unique(self._targets))

    def _preprocess_initial_data(self, data):
        return data

    def set_classes_range(self, low=0, high=None):
        self._low_range = low
        self._high_range = high

        if low == high:
            high = high + 1

        classes = self.classes_order[low:high]
        idxes = np.where(np.isin(self._targets, classes))[0]

        self._mapping = {fake_idx: real_idx for fake_idx, real_idx in enumerate(idxes)}
        if low != high - 1:
            self._update_memory_mapping()

    def set_idxes(self, idxes):
        self._mapping = {fake_idx: real_idx for fake_idx, real_idx in enumerate(idxes)}

    def _update_memory_mapping(self):
        if len(self._memory_idxes):
            examplars_mapping = {
                fake_idx: real_idx
                for fake_idx, real_idx in zip(
                    range(len(self._mapping), len(self._memory_idxes)+len(self._mapping)),
                    self._memory_idxes
                )
            }
            for k, v in examplars_mapping.items():
                assert k not in self._mapping
                self._mapping[k] = v

    def set_memory(self, idxes):
        print("Setting {} memory examplars.".format(len(idxes)))
        self._memory_idxes = idxes

    def get_true_index(self, fake_idx):
        return self._mapping[fake_idx]

    def __len__(self):
        return len(self._mapping)

    def __getitem__(self, idx):
        real_idx = self._mapping[idx]
        x, real_y = self._data[real_idx], self._targets[real_idx]

        x = Image.fromarray(x)
        x = self._transforms(x)

        y = np.where(self.classes_order == real_y)[0][0]

        return (real_idx, idx), x, y


class iCIFAR10(IncrementalDataset):
    _base_dataset = datasets.cifar.CIFAR10
    _train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    _common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(iCIFAR10):
    _base_dataset = datasets.cifar.CIFAR100
    _common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]


class iMNIST(IncrementalDataset):
    _base_dataset = datasets.MNIST
    _train_transforms = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    _common_transforms = [
        transforms.ToTensor()
    ]


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
