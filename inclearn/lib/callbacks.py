import copy

import torch


class Callback:
    def __init__(self):
        self._iteration = 0
        self._in_training = True

    @property
    def in_training(self):
        return self._in_training

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, metric=None):
        self._iteration += 1

    def before_step(self):
        pass


class GaussianNoiseAnnealing(Callback):
    """Add gaussian noise to the gradients.

    Add gaussian noise to the gradients with the given mean & std. The std will
    decrease at each batch up to 0.

    # References:
    - Adding Gradient Noise Improves Learning for Very Deep Networks
    - https://arxiv.org/abs/1511.06807

    :param eta: TODO
    :param gamma: Decay rate.
    """
    def __init__(self, parameters, eta=0.3, gamma=0.55):
        self._parameters = parameters
        self._eta = eta
        self._gamma = gamma

        super(GaussianNoiseAnnealing, self).__init__()

    def before_step(self):
        variance = self._eta / ((1 + self._iteration) ** self._gamma)

        for param in self._parameters:
            # Noise on gradients:
            noise = torch.randn(param.grad.shape, device=param.grad.device) * variance
            param.grad.add_(noise)


class EarlyStopping(Callback):
    def __init__(self, network, minimize_metric=True, patience=5, epsilon=1e-3):
        self._patience = patience
        self._wait = 0

        if minimize_metric:
            self._cmp_fun = lambda old, new: (old - epsilon) > new
            self._best = float('inf')
        else:
            self._cmp_fun = lambda old, new: (old + epsilon) < new
            self._best = float("-inf")

        self.network = network

        self._record = []

        super(EarlyStopping, self).__init__()

    def on_epoch_end(self, metric):
        self._record.append(metric)

        if self._cmp_fun(self._best, metric):
            self._best = metric
            self._wait = 0

            self.network = copy.deepcopy(self.network)
        else:
            self._wait += 1
            if self._wait == self._patience:
                print("Early stopping, metric is: {}.".format(metric))
                print(self._record[-self._patience:])
                self._in_training = False

        super(EarlyStopping, self).on_epoch_end(metric=metric)
