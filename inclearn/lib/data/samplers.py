import numpy as np
from torch.utils.data.sampler import BatchSampler


class MemoryOverSampler(BatchSampler):

    def __init__(self, y, memory_flags, batch_size=128, **kwargs):
        self.indexes = self._oversample(y, memory_flags)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __iter__(self):
        np.random.shuffle(self.indexes)

        for batch_index in range(len(self)):
            low_index = batch_index * self.batch_size
            high_index = (batch_index + 1) * self.batch_size

            yield self.indexes[low_index:high_index].tolist()

    def _oversample(self, y, memory_flags):
        old_indexes = np.where(memory_flags == 1.)[0]
        new_indexes = np.where(memory_flags == 0.)[0]

        old, new = y[old_indexes], y[new_indexes]

        old_qt = self._mean_quantity(old)
        new_qt = self._mean_quantity(new)

        assert new_qt > old_qt, (new_qt, old_qt)
        factor = new_qt / old_qt

        indexes = [np.where(memory_flags == 0)[0]]
        for class_id in np.unique(y):
            indexes.append(np.repeat(np.where(old == class_id)[0], factor))

        indexes = np.concatenate(indexes)
        return indexes

    @staticmethod
    def _mean_quantity(y):
        return np.mean(np.bincount(y))


class MultiSampler(BatchSampler):
    """Sample same batch several times. Every time it's a little bit different
    due to data augmentation. To be used with ensembling models."""

    def __init__(self, nb_samples, batch_size, factor=1, **kwargs):
        self.nb_samples = nb_samples
        self.factor = factor
        self.batch_size = batch_size

    def __len__(self):
        return len(self.y) / self.batch_size

    def __iter__(self):
        pass


class TripletCKSampler(BatchSampler):
    """Samples positives pair that will be then be mixed in triplets.

    C = number of classes
    K = number of instances per class

    References:
        * Facenet: A unified embedding for face recognition and clustering
          Schroff et al.
          CVPR 2015.
    """

    def __init__(self, y, nb_per_class=4, nb_classes=20):
        assert len(np.unique(y)) >= nb_classes

        self.y = y
        self.nb_per_class = nb_per_class
        self.nb_classes = nb_classes

        self._classes = np.unique(y)
        self._class_to_indexes = {
            class_idx: np.where(y == class_idx)[0] for class_idx in self._classes
        }

    def __len__(self):
        return len(self.y) // (self.nb_per_class * self.nb_classes)

    def __iter__(self):
        for _ in range(len(self)):
            indexes = []

            classes = np.random.choice(self._classes, size=self.nb_classes, replace=False)
            for class_id in classes:
                class_indexes = np.random.choice(
                    self._class_to_indexes[class_id],
                    size=self.nb_per_class,
                    replace=bool(len(self._class_to_indexes[class_id]) < self.nb_per_class)
                )

                indexes.extend(class_indexes.tolist())

            yield indexes


class TripletSampler(BatchSampler):
    """Samples elements so that each batch is constitued by a third of anchor, a third
    of positive, and a third of negative.

    Reference:
        * Openface: A general-purpose face recognition library with mobile applications.
          Amos et al.
          2016
     """

    def __init__(self, y, batch_size=128):
        self.y = y
        self.batch_size = (batch_size // 3)
        print("Triplet Sampler has a batch size of {}.".format(3 * self.batch_size))

        self._classes = set(np.unique(y).tolist())
        self._class_to_indexes = {
            class_idx: np.where(y == class_idx)[0] for class_idx in self._classes
        }
        self._indexes = np.arange(len(y))

    def __len__(self):
        return len(self.y) // self.batch_size

    def __iter__(self):
        self._random_permut()

        for batch_index in range(len(self)):
            indexes = []

            for i in range(self.batch_size):
                anchor_index = self._indexes[batch_index * i]
                anchor_class = self.y[batch_index * i]

                pos_index = anchor_index
                while pos_index == anchor_index:
                    pos_index = np.random.choice(self._class_to_indexes[anchor_class])

                neg_class = np.random.choice(list(self._classes - set([anchor_class])))
                neg_index = np.random.choice(self._class_to_indexes[neg_class])

                indexes.append(anchor_index)
                indexes.append(pos_index)
                indexes.append(neg_index)

            yield indexes

    def _random_permut(self):
        shuffled_indexes = np.random.permutation(len(self.y))
        self.y = self.y[shuffled_indexes]
        self._indexes = self._indexes[shuffled_indexes]


class NPairSampler(BatchSampler):

    def __init__(self, y, n_classes=10, n_samples=2, **kwargs):
        self.y = y
        self.n_classes = n_classes
        self.n_samples = n_samples

        self._classes = np.sort(np.unique(y))
        self._distribution = np.bincount(y) / np.bincount(y).sum()
        self._batch_size = self.n_samples * self.n_classes

        self._class_to_indexes = {
            class_index: np.where(y == class_index)[0] for class_index in self._classes
        }

        self._class_counter = {class_index: 0 for class_index in self._classes}

    def __iter__(self):
        for indexes in self._class_to_indexes.values():
            np.random.shuffle(indexes)

        count = 0
        while count + self._batch_size < len(self.y):
            classes = np.random.choice(
                self._classes, self.n_classes, replace=False, p=self._distribution
            )
            batch_indexes = []

            for class_index in classes:
                class_counter = self._class_counter[class_index]
                class_indexes = self._class_to_indexes[class_index]

                class_batch_indexes = class_indexes[class_counter:class_counter + self.n_samples]
                batch_indexes.extend(class_batch_indexes)

                self._class_counter[class_index] += self.n_samples

                if self._class_counter[class_index] + self.n_samples > len(
                    self._class_to_indexes[class_index]
                ):
                    np.random.shuffle(self._class_to_indexes[class_index])
                    self._class_counter[class_index] = 0

            yield batch_indexes

            count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.y) // self._batch_size
