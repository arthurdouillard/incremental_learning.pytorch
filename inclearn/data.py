import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from PIL import Image


class IncrementalDataset(torch.utils.data.Dataset):
    _base_dataset = None
    _train_transforms = []
    _common_transforms = [transforms.ToTensor()]

    def __init__(self, data_path="data", train=True, randomize_class=False,
                 increment=10):
        self._train = train
        self._increment = increment

        dataset = self._base_dataset(
            data_path,
            train=train,
            download=True,
        )
        self._data = self._preprocess_initial_data(dataset.data)
        self._targets = np.array(dataset.targets)

        trsf = self._train_transforms if train else []
        trsf = trsf + self._common_transforms
        self._transforms = transforms.Compose(trsf)

        self.set_classes_range(0, self._increment)

    @property
    def total_n_classes(self):
        return len(np.unique(self._targets))

    def _preprocess_initial_data(self, data):
        return data

    def set_classes_range(self, low=0, high=None):
        self._low_range = low
        self._high_range = high

        if low != high:
            idxes = np.where(
                np.logical_and(self._targets >= low, self._targets < high)
            )[0]
        else:
            idxes = np.where(self._targets == low)[0]

        self._mapping = {fake_idx: real_idx for fake_idx, real_idx in enumerate(idxes)}

    def set_examplars(self, idxes):
        examplars_mapping = {
            fake_idx: real_idx
            for fake_idx, real_idx in zip(range(len(self._mapping), len(idxes)), idxes)
        }
        for k in examplars_mapping:
            assert k not in self._mapping
        self._mapping.update(examplars_mapping)

    def get_true_index(self, fake_idx):
        return self._mapping[fake_idx]

    def __len__(self):
        return len(self._mapping)

    def __getitem__(self, idx):
        real_idx = self._mapping[idx]
        x, y = self._data[real_idx], self._targets[real_idx]

        x = Image.fromarray(x)
        x = self._transforms(x)
        return (real_idx, idx), x, y


class iCIFAR10(IncrementalDataset):
    _base_dataset = datasets.cifar.CIFAR10
    _train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    _common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]


class iCIFAR100(iCIFAR10):
    _base_dataset = datasets.cifar.CIFAR100


class iMNIST(IncrementalDataset):
    _base_dataset = datasets.MNIST
    _train_transforms = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
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
