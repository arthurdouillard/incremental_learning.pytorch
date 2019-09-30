import torch
import torch.nn as nn
import torch.nn.functional as F

from inclearn.lib import pooling


class DownsampleStride(nn.Module):

    def __init__(self, n=2):
        super(DownsampleStride, self).__init__()
        self._n = n

    def forward(self, x):
        x = x[..., ::2, ::2]
        return torch.cat((x, x.mul(0)), 1)


class DownsampleConv(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride=2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        return self.conv(x)


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, final_relu=False, increase_dim=False, downsampling="stride"
    ):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, 2 if increase_dim else 1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if increase_dim:
            if downsampling == "stride":
                self.shortcut = DownsampleStride()
            elif downsampling == "conv":
                self.shortcut = DownsampleConv(inplanes, planes)
            else:
                raise ValueError("Unknown downsampler {}.".format(downsampling))
        else:
            self.shortcut = lambda x: x

        self.final_relu = nn.ReLU(inplace=True) if final_relu else lambda x: x

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))

        x = self.shortcut(x)

        y = x + y
        y = self.final_relu(y)

        return y


class PreActResidualBlock(nn.Module):
    """ResNet v2 version of the residual block.

    Instead of the order conv->bn->relu we use bn->relu->conv.
    """
    expansion = 1

    def __init__(
        self, inplanes, planes, final_relu=False, increase_dim=False, downsampling="stride"
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, 2 if increase_dim else 1)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if increase_dim:
            if downsampling == "stride":
                self.shortcut = DownsampleStride()
            elif downsampling == "conv":
                self.shortcut = DownsampleConv(inplanes, planes)
            else:
                raise ValueError("Unknown downsampler {}.".format(downsampling))
        else:
            self.shortcut = lambda x: x

        self.final_relu = nn.ReLU(inplace=True) if final_relu else lambda x: x

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        y = self.conv2(F.relu(self.bn2(y), inplace=True))

        shortcut = self.shortcut(x)
        y = shortcut + y

        return y


class ResNet(nn.Module):

    def __init__(
        self,
        block_sizes,
        nf=16,
        channels=3,
        preact=False,
        zero_residual=False,
        pooling_config={"type": "avg"},
        downsampling="stride",
        block_relu=False
    ):
        super().__init__()

        self._downsampling_type = downsampling
        self._block_relu = block_relu

        Block = ResidualBlock if not preact else PreActResidualBlock

        self.conv_1_3x3 = conv3x3(channels, nf)
        self.bn_1 = nn.BatchNorm2d(nf)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = nf
        self.stages = nn.ModuleList(
            [
                self._make_layer(Block, 1 * nf, 1 * nf, block_sizes[0], stride=1),
                self._make_layer(Block, 1 * nf, 2 * nf, block_sizes[1], stride=2),
                self._make_layer(Block, 2 * nf, 4 * nf, block_sizes[2], stride=2),
                self._make_layer(Block, 4 * nf, 8 * nf, block_sizes[3], stride=2, last=True)
            ]
        )

        if pooling_config["type"] == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_config["type"] == "weldon":
            self.pool = pooling.WeldonPool2d(**pooling_config)
        else:
            raise ValueError("Unknown pooling type {}.".format(pooling_config["type"]))

        self.out_dim = 8 * nf

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, Block, inplanes, planes, block_size, stride=1, last=False):
        layers = []

        if stride != 1:
            layers.append(
                Block(
                    inplanes,
                    planes,
                    increase_dim=True,
                    downsampling=self._downsampling_type,
                    final_relu=self._block_relu
                )
            )
        else:
            layers.append(Block(inplanes, planes, final_relu=self._block_relu))

        for i in range(1, block_size):
            if last and i == block_size - 1:
                final_relu = False
            else:
                final_relu = self._block_relu
            layers.append(Block(planes, planes, final_relu=final_relu))

        return nn.Sequential(*layers)

    def forward(self, x, attention_hook=False):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.maxpool(x)

        intermediary_features = []
        for stage in self.stages:
            x = stage(x)
            intermediary_features.append(x)

        raw_features = self.end_features(x)
        features = self.end_features(F.relu(x, inplace=False))

        if attention_hook:
            return raw_features, features, intermediary_features
        return raw_features, features

    def end_features(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18(**kwargs):
    return ResNet([2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet([3, 4, 6, 3], **kwargs)
