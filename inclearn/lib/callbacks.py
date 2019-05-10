import torch


class GaussianNoiseAnnealing:
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

        self._iteration = 0

    def add_noise(self):
        variance = self._eta / ((1 + self._iteration) ** self._gamma)

        for param in self._parameters:
            # L2 regularization on gradients
            param.grad.add_(0.0001, torch.norm(param.grad, p=2))

            # Noise on gradients:
            noise = torch.randn(param.grad.shape, device=param.grad.device) * variance
            param.grad.add_(noise)

            param.grad.clamp_(min=-5, max=5)

    def step(self):
        self._iteration += 1
