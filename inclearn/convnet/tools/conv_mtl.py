##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/pytorch/pytorch
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##
## Modified by: Arthur Douillard in order to apply to Incremental Learning
## Quoting Yaoyao Liu:
##      For the first incremental phase, we train the network without SS weights.
##      For the second incremental phase, we initialize SS weights with ones and
##      zeros as MTL, we update SS weights and keep the network frozen. For the
##      following incremental phases, at the beginning of each incremental phase,
##      we apply the SS weights learned last phase to the frozen network and get
##      a new network. Then we freeze the new network, reset SS weights and update
##      SS weights during training;
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" MTL CONV layers. """
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class Conv2dMtl(Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_channels = self.weight.shape[0]
        in_channels = self.weight.shape[1]

        self.mtl_weight = Parameter(torch.ones(out_channels, in_channels, 1, 1))
        self.mtl_bias = Parameter(torch.zeros(out_channels))

        self._apply_mtl = False
        self._apply_mtl_bias = False
        self._apply_bias_on_weights = False
        self.reset_mtl_parameters()

    def conv2d_forward(self, input, weight):
        if self.apply_mtl:
            weight = self.weight.mul(self.mtl_weight.expand(self.weight.shape))
            if self.apply_bias_on_weights:
                weight = weight.add(self.mtl_bias[..., None, None, None].expand(self.weight.shape))
        else:
            weight = self.weight

        if self.bias and self.apply_mtl and not self.apply_bias_on_weights:
            bias = self.bias + self.mtl_bias
        elif self.apply_mtl_bias and not self.apply_bias_on_weights:
            bias = self.mtl_bias
        else:
            bias = self.bias

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def reset_mtl_parameters(self):
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.mtl_bias.data.uniform_(0, 0)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @property
    def apply_mtl(self):
        return self._apply_mtl

    @property
    def apply_mtl_bias(self):
        return self._apply_mtl_bias

    @apply_mtl.setter
    def apply_mtl(self, b):
        assert isinstance(b, bool), b
        self._apply_mtl = b

    @apply_mtl_bias.setter
    def apply_mtl_bias(self, b):
        assert isinstance(b, bool), b
        self._apply_mtl_bias = b

    @property
    def apply_bias_on_weights(self):
        return self._apply_bias_on_weights

    @apply_bias_on_weights.setter
    def apply_bias_on_weights(self, b):
        assert isinstance(b, bool), b
        self._apply_bias_on_weights = b

    def freeze_convnet(self, freeze):
        self.weight.requires_grad = not freeze
        if self.bias:
            self.bias.requires_grad = not freeze

    def fuse_mtl_weights(self):
        with torch.no_grad():
            new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
            self.weight.mul_(new_mtl_weight)

            if self.apply_bias_on_weights:
                self.weight.add_(self.mtl_bias[..., None, None, None].expand(self.weight.shape))
            elif self.bias:
                self.bias.add_(self.mtl_bias)
