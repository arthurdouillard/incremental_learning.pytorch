import pickle

import gensim
import numpy as np
import torch
from torch import nn

from inclearn.lib.data import fetch_word_embeddings


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
        freeze_embedding=True
    ):
        super().__init__()

        path = fetch_word_embeddings("/data/douillard/", embeddings)
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        if dataset == "cifar100":
            weights, labels = _get_cifar100_embeddings(gensim_model)
        else:
            raise ValueError("Unknown dataset {} for word embeddings.".format(dataset))

        del gensim_model
        self.emb = torch.nn.Embedding(
            num_embeddings=weights.shape[0], embedding_dim=weights.shape[1]
        )
        self.emb.weight = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=not freeze_embedding
        )

        if mlp_dims is not None:
            self.mlp = MLP(
                input_dim=noise_dimension + weights.shape[1],
                hidden_dims=mlp_dims,
                use_bn=use_bn,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout
            )
        else:
            self.mlp = None

        self.noise_dimension = noise_dimension
        self.to(device)

    def forward(self, x):
        word = self.emb(x)
        noise = torch.randn(len(x), self.noise_dimension).to(word.device)

        if self.mlp:
            return self.mlp(torch.cat((word, noise), dim=-1))
        return x


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, use_bn=True, input_dropout=0., hidden_dropout=0.):
        super().__init__()

        layers = []
        for index, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, dim, bias=not use_bn))
            nn.init.normal_(layers[-1].weight, std=0.02)

            if input_dropout and index == 0:
                layers.append(nn.Dropout(p=input_dropout, inplace=True))
            elif hidden_dropout and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=hidden_dropout, inplace=True))

            if index < len(hidden_dims) - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            input_dim = dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Prepare embeddings for dataset
def _get_cifar100_embeddings(gensim_model):
    with open("/data/douillard/cifar-100-python/meta", "rb") as f:
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
