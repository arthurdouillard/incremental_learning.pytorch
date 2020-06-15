import torch
from torch import nn
from torch.nn import functional as F


class MemoryBank:

    def __init__(self, device, momentum=0.5):
        self.features = None
        self.targets = None

        self.momentum = momentum

        self.device = device

    def add(self, features, targets):
        if self.features is None:
            self.features = features
            self.targets = targets
        else:
            self.features = torch.cat((self.features, features.to(self.device)), dim=0)
            self.targets = torch.cat((self.targets, targets.to(self.device)), dim=0)

    def get(self, indexes):
        return self.features[indexes]

    def get_neg(self, indexes, n=10):
        neg_indexes = torch.ones(len(self.features)).bool()
        neg_indexes[indexes] = False
        nb = min(n, len(self.features) - len(indexes))
        rnd_indexes = torch.multinomial(torch.ones(nb), nb)

        return self.features[neg_indexes][rnd_indexes]

    def update(self, features, indexes):
        self.features[indexes] = self.momentum * self.features[indexes]\
                                 + (1 - self.momentum * features)
