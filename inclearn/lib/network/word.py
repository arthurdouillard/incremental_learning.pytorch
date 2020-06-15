import logging
import os
import pickle

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F

import gensim
from inclearn.lib.data import fetch_word_embeddings

from .mlp import MLP

logger = logging.getLogger(__name__)


class Word2vec(nn.Module):

    def __init__(
        self,
        embeddings="googlenews",
        dataset="cifar100",
        mlp_dims=None,
        use_bn=True,
        input_dropout=0.2,
        hidden_dropout=0.5,
        device=None,
        noise_dimension=50,
        noise_type="normal",
        freeze_embedding=True,
        scale_embedding=None,
        data_path=None
    ):
        super().__init__()

        self.emb, _ = get_embeddings(dataset, embeddings, frozen=freeze_embedding, path=data_path)
        if isinstance(scale_embedding, list):
            logger.info(f"Scaling semantic embedding in {scale_embedding}.")
            self.emb.weight.data = Scaler(scale_embedding).fit_transform(self.emb.weight.data)
        elif isinstance(scale_embedding, str) and scale_embedding == "l2":
            self.emb.weight.data = F.normalize(self.emb.weight.data, dim=-1, p=2)

        semantic_dim = self.emb.weight.shape[1]
        logger.info(f"Semantic dimension: {semantic_dim}.")

        if mlp_dims is not None:
            self.mlp = MLP(
                input_dim=noise_dimension + semantic_dim,
                hidden_dims=mlp_dims,
                use_bn=use_bn,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout
            )
        else:
            self.mlp = None

        self.noise_dimension = noise_dimension
        self.noise_type = noise_type
        self.to(device)
        self.device = device
        self.out_dim = mlp_dims[-1]

        self.linear_transform = None

    def add_linear_transform(self, bias=False):
        self.linear_transform = nn.Linear(self.out_dim, self.out_dim, bias=bias)
        #self.linear_transform.weight.data = torch.eye(self.out_dim)
        #self.linear_transform.weight.data += torch.empty(self.out_dim, self.out_dim).normal_(mean=0, std=0.1)
        if bias:
            self.linear_transform.bias.data.fill_(0.)
        self.linear_transform.to(self.device)

    def forward(self, x, only_word=False):
        word = self.emb(x)

        if only_word:
            return word

        if self.noise_dimension:
            if self.noise_type == "normal":
                noise = torch.randn(len(x), self.noise_dimension).to(word.device)
            elif self.noise_type == "uniform":
                noise = torch.rand(len(x), self.noise_dimension).to(word.device)
            else:
                raise ValueError(f"Unknown noise type {self.noise_type}.")

        if self.mlp:
            fake_features = self.mlp(torch.cat((word, noise), dim=-1))
            if self.linear_transform:
                fake_features = self.linear_transform(fake_features)
            return fake_features

        return word


def get_embeddings(dataset, embeddings, path=None, frozen=True):
    if dataset == "cifar100":
        weights, labels = _get_cifar100_embeddings(embeddings, path)
    elif dataset == "awa2_attributes":
        weights, labels = _get_awa2_attributes(path)
    elif dataset == "awa2_attributes_mat":
        weights, labels = _get_awa2_attributes_mat(path)
    elif dataset == "cub200_attributes":
        weights, labels = _get_cub200_attributes(path)
    elif dataset == "cub200_attributes_mat":
        weights, labels = _get_cub200_attributes_mat(path)
    elif dataset == "apy_attributes_mat":
        weights, labels = _get_apy_attributes_mat(path)
    elif dataset == "w2v_wiki_imagenet100ucir_300d":
        weights = np.load(os.path.join(path, "word2vec_wiki_imagenet100ucir_300d.npy"))
        labels = np.load(os.path.join(path, "word2vec_wiki_imagenet100ucir_300d_labels.npy"))
    elif dataset == "w2v_wiki_imagenet100_300d":
        weights = np.load(os.path.join(path, "word2vec_wiki_imagenet100_300d.npy"))
        labels = np.load(os.path.join(path, "word2vec_wiki_imagenet100_300d_labels.npy"))
    elif dataset == "w2v_wiki_imagenet100ucir_500d":
        weights = np.load(os.path.join(path, "word2vec_wiki_imagenet100ucir_500d.npy"))
        labels = np.load(os.path.join(path, "word2vec_wiki_imagenet100ucir_500d_labels.npy"))
    elif dataset == "w2v_wiki_imagenet100_500d":
        weights = np.load(os.path.join(path, "word2vec_wiki_imagenet100_500d.npy"))
        labels = np.load(os.path.join(path, "word2vec_wiki_imagenet100_500d_labels.npy"))
    elif dataset == "lad_attributes":
        weights, labels = _get_lad_attributes(path)
    else:
        raise ValueError("Unknown dataset {} for word embeddings.".format(dataset))

    emb = torch.nn.Embedding(num_embeddings=weights.shape[0], embedding_dim=weights.shape[1])
    emb.weight = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=not frozen)

    return emb, labels


# Prepare embeddings for dataset
def _get_cifar100_embeddings(embeddings_type, data_path=None):
    if data_path is None:
        data_path = "/data/douillard/"
    path = fetch_word_embeddings(data_path, embeddings_type)
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    with open(os.path.join(data_path, "cifar-100-python/meta"), "rb") as f:
        meta = pickle.load(f)
    labels = meta["fine_label_names"]

    fixed_labels = {
        "aquarium_fish": "fish",
        "lawn_mower": "lawnmower",
        "maple_tree": "maple",
        "oak_tree": "oak",
        "palm_tree": "palm",
        "pickup_truck": "pickup",
        "pine_tree": "pine",
        "sweet_pepper": "pepper",
        "willow_tree": "willow"
    }
    for missing, replacement in fixed_labels.items():
        labels[labels.index(missing)] = replacement

    trimmed_weights = np.empty((len(labels), gensim_model.vectors.shape[1]), dtype=np.float32)
    for i, label in enumerate(labels):
        index = gensim_model.index2word.index(label)
        trimmed_weights[i] = gensim_model.vectors[index]

    return trimmed_weights, labels


def _get_awa2_attributes(path=None):
    if path is None:
        path = "/data/douillard/"
    attributes = np.loadtxt(
        os.path.join(path, "awa2/Animals_with_Attributes2/predicate-matrix-continuous.txt")
    )
    attributes = attributes / np.linalg.norm(attributes, axis=1, keepdims=True)

    labels = []
    with open(os.path.join(path, "awa2/Animals_with_Attributes2/classes.txt")) as f:
        for line in f:
            labels.append(line.strip())

    return attributes, labels


def _get_awa2_attributes_mat(path=None):
    if path is None:
        path = "/data/douillard/"
    attributes = loadmat(os.path.join(path, "zeroshot_split/xlsa17/data/AWA2/att_splits.mat"))
    attributes = attributes["att"].T
    labels = []
    with open(os.path.join(path, "awa2/Animals_with_Attributes2/classes.txt")) as f:
        for line in f:
            labels.append(line.strip())

    return attributes, labels


def _get_cub200_attributes(path=None):
    if path is None:
        path = "/data/douillard/"
    attributes = np.loadtxt(
        os.path.join(path, "CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
    )
    attributes = attributes / np.linalg.norm(attributes, axis=1, keepdims=True)

    labels = []
    with open(os.path.join(path, "CUB_200_2011/classes.txt")) as f:
        for line in f:
            labels.append(line.strip())

    return attributes, labels


def _get_cub200_attributes_mat(path=None):
    if path is None:
        path = "/data/douillard/"
    attributes = loadmat(os.path.join(path, "zeroshot_split/xlsa17/data/CUB/att_splits.mat"))
    attributes = attributes["att"].T

    labels = []
    with open(os.path.join(path, "CUB_200_2011/classes.txt")) as f:
        for line in f:
            labels.append(line.strip())

    return attributes, labels


def _get_apy_attributes_mat(path=None):
    if path is None:
        path = "/data/douillard/"
    attributes = loadmat(os.path.join(path, "zeroshot_split/xlsa17/data/APY/att_splits.mat"))
    attributes = attributes["att"].T

    return attributes, None


def _get_lad_attributes(path=None):
    if path is None:
        path = "/data/douillard/"

    attributes = []
    labels = []
    with open(os.path.join(path, "LAD/attributes_per_class.txt")) as f:
        for line in f:
            line = line.strip().split(", ")
            label = line[0]
            att = line[1]

            att = list(map(float, filter(lambda x: len(x) > 0, att[3:-3].split(" "))))
            attributes.append(np.array(att))
            labels.append(label)

    labels = np.array(labels)
    attributes = np.stack(attributes)

    return attributes, labels


class Scaler:
    """
    Transforms each channel to the range [a, b].
    """

    def __init__(self, feature_range):
        self.feature_range = feature_range

    def fit(self, tensor):
        data_min = torch.min(tensor, dim=0)[0]
        data_max = torch.max(tensor, dim=0)[0]
        data_range = data_max - data_min

        # Handle null values
        data_range[data_range == 0.] = 1.

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        return self

    def transform(self, tensor):
        return tensor.mul_(self.scale_).add_(self.min_)

    def inverse_transform(self, tensor):
        return tensor.sub_(self.min_).div_(self.scale_)

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)
