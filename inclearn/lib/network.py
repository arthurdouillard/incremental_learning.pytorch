import copy

import torch
from torch import nn

from inclearn.lib import factory


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        use_bias=False,
        init="kaiming",
        use_multi_fc=False,
        cosine_similarity=False,
        scaling_factor=False,
        device=None,
        return_features=False
    ):
        super(BasicNet, self).__init__()

        if scaling_factor:
            self.post_processor = LearnedScaler()
        else:
            self.post_processor = None

        self.convnet = factory.get_convnet(convnet_type, nf=64, zero_init_residual=True)

        if cosine_similarity:
            self.classifier = CosineClassifier(self.convnet.out_dim, device)
        else:
            self.classifier = Classifier(self.convnet.out_dim, use_bias, use_multi_fc, init, device)

        self.return_features = return_features
        self.device = device

        self.to(self.device)

    def forward(self, x):
        features = self.convnet(x)
        logits = self.classifier(features)

        if self.return_features:
            return features, logits
        return logits

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def add_classes(self, n_classes):
        self.classifier.add_classes(n_classes)

    def extract(self, x):
        return self.convnet(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes


class Classifier(nn.Module):
    def __init__(self, features_dim, use_bias, use_multi_fc, init, device):
        super().__init__()

        self.features_dim = features_dim
        self.use_bias = use_bias
        self.use_multi_fc = use_multi_fc
        self.init = init
        self.device = device

        self.n_classes = 0

        self.classifier = None

    def forward(self, features):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.use_multi_fc:
            logits = []
            for classifier in self.classifier:
                logits.append(classifier(features))
            logits = torch.cat(logits, 1)
        else:
            logits = self.classifier(features)

        return logits

    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = nn.ModuleList([])

        new_classifier = self._gen_classifier(n_classes)
        self.classifier.append(new_classifier)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
        classifier = nn.Linear(self.features_dim, n_classes, bias=self.use_bias).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier


class CosineClassifier(nn.Module):
    def __init__(self, features_dim, device):
        super().__init__()

        self.n_classes = 0
        self.weights = None
        self.features_dim = features_dim

        self.device = device

    def forward(self, features):
        features_norm = features / (features.norm(dim=1)[:, None] + 1e-8)
        weights_norm = self.weights / (self.weights.norm(dim=1)[:, None] + 1e-8)

        similarities = torch.mm(features_norm, weights_norm.transpose(0, 1))

        return similarities

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(self.n_classes + n_classes, self.features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        if self.weights is not None:
            new_weights.data[:self.n_classes] = copy.deepcopy(self.weights.data)

        del self.weights
        self.weights = new_weights
        self.to(self.device)
        self.n_classes += n_classes


class LearnedScaler(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, inputs):
        return self.scale * inputs


# -------------
# Recalibration
# -------------

class CalibrationWrapper(nn.Module):
    """Wraps several calibration models, each being applied on different targets."""

    def __init__(self):
        super().__init__()

        self.start_indexes = []
        self.end_indexes = []
        self.models = nn.ModuleList([])

    def add_model(self, model, start_index, end_index):
        """Adds a calibration model that will applies on target between the two indexes.

        The models must be added in the right targets order!
        """
        self.models.append(model)
        self.start_indexes.append(start_index)
        self.end_indexes.append(end_index)

    def forward(self, inputs):
        corrected_inputs = []

        if self.start_indexes[0] != 0:
            corrected_inputs.append(inputs[..., :self.start_indexes[0]])

        for model, start_index, end_index in zip(self.models, self.start_indexes, self.end_indexes):
            corrected_inputs.append(model(inputs[..., start_index:end_index]))

        if self.end_indexes[-1] != inputs.shape[1]:
            corrected_inputs.append(inputs[..., self.end_indexes[-1]:])

        corrected_inputs = torch.cat(corrected_inputs, dim=-1)

        return corrected_inputs


class LinearModel(nn.Module):
    """Linear model applying on the logits alpha * x + beta.

    By default, this model is initialized as an identity operation.

    See https://arxiv.org/abs/1905.13260 for an example usage.

    :param alpha: A learned scalar.
    :param beta: A learned scalar.
    """

    def __init__(self, alpha=1., beta=0.):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, inputs):
        return self.alpha * inputs + self.beta


class TemperatureScaling(nn.Module):
    """Applies a learned temperature on the logits.

    See https://arxiv.org/abs/1706.04599.
    """

    def __init__(self, temperature=1):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, inputs):
        return inputs / self.temperature
