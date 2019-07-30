import numpy as np
from torch.utils.data.sampler import BatchSampler


class NPairSampler(BatchSampler):
    def __init__(self, y, n_classes=10, n_samples=2):
        self.y = y
        self.n_classes = n_classes
        self.n_samples = n_samples

        self._classes = np.sort(np.unique(y))
        self._distribution = np.bincount(y) / np.bincount(y).sum()
        self._batch_size = self.n_samples * self.n_classes

        self._class_to_indexes = {class_index: np.where(y == class_index)[0]
                                  for class_index in self._classes}

        self._class_counter = {class_index: 0 for class_index in self._classes}

    def __iter__(self):
        for indexes in self._class_to_indexes.values():
            np.random.shuffle(indexes)

        count = 0
        while count + self._batch_size < len(self.y):
            classes = np.random.choice(self._classes, self.n_classes, replace=False,
                                       p=self._distribution)
            batch_indexes = []

            for class_index in classes:
                class_counter = self._class_counter[class_index]
                class_indexes = self._class_to_indexes[class_index]

                class_batch_indexes = class_indexes[class_counter: class_counter + self.n_samples]
                batch_indexes.extend(class_batch_indexes)

                self._class_counter[class_index] += self.n_samples

                if self._class_counter[class_index] + self.n_samples > len(self._class_to_indexes[class_index]):
                    np.random.shuffle(self._class_to_indexes[class_index])
                    self._class_counter[class_index] = 0

            yield batch_indexes

            count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.y) // self._batch_size
