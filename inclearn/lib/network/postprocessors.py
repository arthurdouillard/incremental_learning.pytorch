import torch
from torch import nn


class ConstantScalar(nn.Module):

    def __init__(self, constant=1., bias=0., **kwargs):
        super().__init__()

        self.factor = constant
        self.bias = bias

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, x):
        if hasattr(self, "bias"):
            return self.factor * x + self.bias
        else:
            return self.factor * x


class FactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwargs):
        super().__init__()

        self.factor = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)


class InvertedFactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwargs):
        super().__init__()

        self._factor = nn.Parameter(torch.tensor(initial_value))

    @property
    def factor(self):
        return 1 / (self._factor + 1e-7)

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)


class HeatedUpScalar(nn.Module):

    def __init__(self, first_value, last_value, nb_steps, scope="task", **kwargs):
        super().__init__()

        self.scope = scope
        self.first_value = first_value
        self.step = (max(first_value, last_value) - min(first_value, last_value)) / (nb_steps - 1)

        if first_value > last_value:
            self._factor = -1
        else:
            self._factor = 1

        self._increment = 0

        print("Heated-up factor is {} with {} scope.".format(self.factor, self.scope))

    def on_task_end(self):
        if self.scope == "task":
            self._increment += 1
        print("Heated-up factor is {}.".format(self.factor))

    def on_epoch_end(self):
        if self.scope == "epoch":
            self._increment += 1

    @property
    def factor(self):
        return self.first_value + (self._factor * self._increment * self.step)

    def forward(self, inputs):
        return self.factor * inputs
