import collections
import glob
import logging
import math
import os
import warnings

import numpy as np
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")


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

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)


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


class ImageNet100(DataHandler):
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    imagenet_size = 100
    open_image = True
    suffix = ""
    metadata_path = None

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)

    def base_dataset(self, data_path, train=True, download=False):
        if download:
            warnings.warn(
                "ImageNet incremental dataset cannot download itself,"
                " please see the instructions in the README."
            )

        split = "train" if train else "val"

        print("Loading metadata of ImageNet_{} ({} split).".format(self.imagenet_size, split))
        metadata_path = os.path.join(
            data_path if self.metadata_path is None else self.metadata_path,
            "{}_{}{}.txt".format(split, self.imagenet_size, self.suffix)
        )

        self.data, self.targets = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")

                self.data.append(os.path.join(data_path, path))
                self.targets.append(int(target))

        self.data = np.array(self.data)

        return self


class ImageNet100UCIR(ImageNet100):
    suffix = "_ucir"


class ImageNet1000(ImageNet100):
    imagenet_size = 1000


class TinyImageNet200(DataHandler):
    train_transforms = [
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_transforms = [transforms.Resize(64)]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    open_image = True

    class_order = list(range(200))

    def set_custom_transforms(self, transforms_dict):
        if not transforms_dict.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)
        if transforms_dict.get("crop"):
            logger.info("Crop with padding of {}".format(transforms_dict.get("crop")))
            self.train_transforms[0] = transforms.RandomCrop(
                64, padding=transforms_dict.get("crop")
            )

    def base_dataset(self, data_path, train=True, download=False):
        if train:
            self._train_dataset(data_path)
        else:
            self._val_dataset(data_path)

        return self

    def _train_dataset(self, data_path):
        self.data, self.targets = [], []

        train_dir = os.path.join(data_path, "train")
        for class_id, class_name in enumerate(os.listdir(train_dir)):
            paths = glob.glob(os.path.join(train_dir, class_name, "images", "*.JPEG"))
            targets = [class_id for _ in range(len(paths))]

            self.data.extend(paths)
            self.targets.extend(targets)

        self.data = np.array(self.data)

    def _val_dataset(self, data_path):
        self.data, self.targets = [], []

        self.classes2id = {
            class_name: class_id
            for class_id, class_name in enumerate(os.listdir(os.path.join(data_path, "train")))
        }
        self.id2classes = {v: k for k, v in self.classes2id.items()}

        with open(os.path.join(data_path, "val", "val_annotations.txt")) as f:
            for line in f:
                split_line = line.split("\t")

                path, class_label = split_line[0], split_line[1]
                class_id = self.classes2id[class_label]

                self.data.append(os.path.join(data_path, "val", "images", path))
                self.targets.append(class_id)

        self.data = np.array(self.data)


class AwA2(DataHandler):
    test_split = 0.2

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.466, 0.459, 0.397], std=[0.211, 0.206, 0.203])
    ]

    open_image = True

    class_order = None

    def _create_class_mapping(self, path):
        label_to_id = {}

        with open(os.path.join(path, "classes.txt"), "r") as f:
            for i, line in enumerate(f.readlines()):
                label_to_id[line.strip().split("\t")[1]] = i

        self.class_order = []
        with open(os.path.join(path, "trainclasses.txt"), "r") as f:
            for i, line in enumerate(f.readlines()):
                self.class_order.append(label_to_id[line.strip()])

        with open(os.path.join(path, "testclasses.txt"), "r") as f:
            for j, line in enumerate(f.readlines(), start=len(self.class_order)):
                self.class_order.append(label_to_id[line.strip()])
        assert len(set(self.class_order)) == len(self.class_order)

        id_to_label = {v: k for k, v in label_to_id.items()}

        return label_to_id, id_to_label

    def set_custom_transforms(self, transforms_dict):
        pass

    def base_dataset(self, data_path, train=True, download=False):
        directory = os.path.join(data_path, "awa2", "Animals_with_Attributes2")
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            pass

        label_to_id, id_to_label = self._create_class_mapping(directory)

        data = collections.defaultdict(list)
        for class_directory in os.listdir(os.path.join(directory, "JPEGImages")):
            class_id = label_to_id[class_directory]

            for image_path in glob.iglob(
                os.path.join(directory, "JPEGImages", class_directory, "*jpg")
            ):
                data[class_id].append(image_path)

        paths = []
        targets = []
        for class_id, class_paths in data.items():
            rnd_state = np.random.RandomState(seed=1)
            indexes = rnd_state.permutation(len(class_paths))

            if train:
                subset = math.floor(len(indexes) * (1 - self.test_split))
                indexes = indexes[:subset]
            else:
                subset = math.ceil(len(indexes) * self.test_split)
                indexes = indexes[subset:]

            paths.append(np.array(class_paths)[indexes])
            targets.extend([class_id for _ in range(len(indexes))])

        self.data = np.concatenate(paths)
        self.targets = np.array(targets)

        self.label_to_id, self.id_to_label = label_to_id, id_to_label

        return self


class CUB200(DataHandler):
    test_split = 0.2

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4836, 0.4921, 0.4243], std=[0.1845, 0.1835, 0.1947])
    ]

    open_image = True

    # from The Good, the bad and the ugly:
    class_order = [
        1, 2, 14, 15, 19, 21, 46, 47, 66, 67, 68, 72, 73, 74, 75, 88, 89, 99,
        148, 149, 0, 13, 33, 34, 100, 119, 109, 84, 7, 53, 170, 40, 55, 108,
        186, 174, 29, 194, 50, 106, 116, 134, 133, 45, 146, 36, 159, 125, 136,
        124, 26, 188, 196, 185, 157, 63, 43, 6, 182, 141, 85, 158, 80, 127,
        10, 144, 28, 165, 58, 94, 154, 9, 140, 101, 78, 105, 191, 4, 82, 177,
        161, 193, 195, 49, 38, 104, 35, 31, 145, 81, 59, 143, 198, 92, 197,
        65, 98, 52, 150, 17, 151, 115, 60, 24, 23, 77, 16, 175, 57, 20, 192,
        56, 39, 152, 87, 12, 117, 120, 178, 61, 153, 91, 37, 139, 181, 95, 171,
        70, 41, 184, 176, 18, 64, 8, 111, 62, 5, 79, 180, 107, 121, 114, 183,
        166, 128, 132, 113, 169, 130, 173,  # seen classes
        42, 110, 22, 97, 54, 129, 138, 122, 155, 123, 199, 71, 172, 27, 118,
        164, 102, 179, 76, 11, 44, 189, 190, 137, 156, 51, 32, 163, 30, 142,
        93, 69, 96, 90, 103, 126, 160, 48, 168, 147, 112, 86, 162, 135, 187,
        83, 25, 3, 131, 167  # unseen classes
    ]  # yapf: disable

    def _create_class_mapping(self, path):
        label_to_id = {}

        self.class_order = []
        with open(os.path.join(path, "classes.txt"), "r") as f:
            for i, line in enumerate(f.readlines()):
                label_to_id[line.strip().split(" ")[1]] = i
                self.class_order.append(i)

        id_to_label = {v: k for k, v in label_to_id.items()}
        return label_to_id, id_to_label

    def set_custom_transforms(self, transforms_dict):
        pass

    def base_dataset(self, data_path, train=True, download=False):
        directory = os.path.join(data_path, "CUB_200_2011")
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            pass

        label_to_id, id_to_label = self._create_class_mapping(directory)

        train_set = set()
        with open(os.path.join(directory, "train_test_split.txt")) as f:
            for line in f:
                line_id, set_id = line.split(" ")
                if int(set_id) == 1:
                    train_set.add(int(line_id))

        c = 1
        data = collections.defaultdict(list)
        for class_directory in sorted(os.listdir(os.path.join(directory, "images"))):
            class_id = label_to_id[class_directory]

            for image_path in sorted(
                os.listdir(os.path.join(directory, "images", class_directory))
            ):
                if not image_path.endswith("jpg"):
                    continue

                image_path = os.path.join(directory, "images", class_directory, image_path)

                if (c in train_set and train) or (c not in train_set and not train):
                    data[class_id].append(image_path)
                c += 1

        self.data, self.targets = self._convert(data)

        self.label_to_id, self.id_to_label = label_to_id, id_to_label

        return self

    @staticmethod
    def _convert(data):
        paths = []
        targets = []
        for class_id, class_paths in data.items():
            paths.extend(class_paths)
            targets.extend([class_id for _ in range(len(class_paths))])

        return np.array(paths), np.array(targets)


class APY(DataHandler):
    test_split = 0.1
    test_max_cap = 100

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3074, 0.2576, 0.2052], std=[0.2272, 0.2147, 0.2105])
    ]

    open_image = True

    # from The Good, the bad and the ugly:
    class_order = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19,  # seen classes from pascal VOC
        20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31  # unseen classes from yahoo
    ]  # yapf: disable

    def _create_class_mapping(self, path):
        label_to_id = {}

        with open(os.path.join(path, "class_names.txt"), "r") as f:
            for i, line in enumerate(f.readlines()):
                label_to_id[line.strip()] = i

        id_to_label = {v: k for k, v in label_to_id.items()}
        return label_to_id, id_to_label

    def set_custom_transforms(self, transforms_dict):
        pass

    def base_dataset(self, data_path, train=True, download=False):
        directory = os.path.join(data_path, "APY")

        label_to_id, id_to_label = self._create_class_mapping(directory)

        paths, targets = [], []
        with open(os.path.join(directory, "data.txt")) as f:
            for line in f:
                p, t = line.split(",")
                paths.append(os.path.join(data_path, p))
                targets.append(label_to_id[t.strip()])

        paths = np.array(paths)
        targets = np.array(targets)

        self.data, self.targets = [], []
        for class_id in np.unique(targets):
            rnd_state = np.random.RandomState(seed=1)

            indexes = np.where(class_id == targets)[0]

            test_amount = int(len(indexes) * self.test_split)
            test_amount = min(test_amount, self.test_max_cap)

            if train:
                amount = len(indexes) - test_amount
            else:
                amount = test_amount

            indexes = rnd_state.choice(indexes, size=amount, replace=False)

            self.data.append(paths[indexes])
            self.targets.append(targets[indexes])

        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

        self.label_to_id, self.id_to_label = label_to_id, id_to_label
        print(f"{len(self.data)} images for {len(self.label_to_id)} classes.")

        return self


class LAD(DataHandler):
    test_split = 0.1

    train_transforms = [
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_transforms = [transforms.Resize((224, 224))]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5815, 0.5567, 0.5078], std=[0.2364, 0.2393, 0.2487])
    ]

    open_image = True

    def _create_class_mapping(self, path):
        label_to_id = {}

        label_to_id = {}
        self.class_order = []
        with open(os.path.join(path, "label_list.txt")) as f:
            for i, line in enumerate(f):
                c = line.strip().split(", ")[1]
                label_to_id[c] = i
                self.class_order.append(i)  # Classes are already in the right order.

        id_to_label = {v: k for k, v in label_to_id.items()}
        return label_to_id, id_to_label

    def set_custom_transforms(self, transforms_dict):
        pass

    def base_dataset(self, data_path, train=True, download=False):
        directory = os.path.join(data_path, "LAD")

        label_to_id, id_to_label = self._create_class_mapping(directory)

        paths = []
        targets = []

        base_path = os.path.join(directory, "images/")
        for class_folder in os.listdir(base_path):
            class_name = "_".join(class_folder.split("_")[1:])
            class_id = label_to_id[class_name]
            class_folder = os.path.join(base_path, class_folder)

            for image_path in glob.iglob(os.path.join(class_folder, "*jpg")):
                paths.append(image_path)
                targets.append(class_id)

        paths = np.array(paths)
        targets = np.array(targets)

        self.data, self.targets = [], []
        for class_id in np.unique(targets):
            rnd_state = np.random.RandomState(seed=1)

            indexes = np.where(class_id == targets)[0]

            if train:
                amount = int(len(indexes) * (1 - self.test_split))
            else:
                amount = int(len(indexes) * self.test_split)

            indexes = rnd_state.choice(indexes, size=amount, replace=False)

            self.data.append(paths[indexes])
            self.targets.append(targets[indexes])

        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

        self.label_to_id, self.id_to_label = label_to_id, id_to_label
        print(f"{len(self.data)} images for {len(self.label_to_id)} classes.")

        return self
