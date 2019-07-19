import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from inclearn.lib import factory, utils


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        convnet_kwargs={},
        use_bias=False,
        init="kaiming",
        use_multi_fc=False,
        cosine_similarity=False,
        scaling_factor=False,
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False
    ):
        super(BasicNet, self).__init__()

        if scaling_factor:
            self.post_processor = LearnedScaler()
        else:
            self.post_processor = None

        self.convnet = factory.get_convnet(convnet_type, **convnet_kwargs)

        self.cosine_similarity = cosine_similarity
        if cosine_similarity:
            self.classifier = CosineClassifier(self.convnet.out_dim, device)
        else:
            self.classifier = Classifier(self.convnet.out_dim, use_bias, use_multi_fc, init, device)

        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.device = device

        if self.extract_no_act:
            print("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            print("No ReLU will be applied on features before feeding the classifier.")

        self.to(self.device)

    def forward(self, x):
        raw_features, features = self.convnet(x)
        logits = self.classifier(raw_features if self.classifier_no_act else features)

        if self.return_features:
            if self.extract_no_act:
                return raw_features, logits
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

    def add_imprinted_classes(self, class_indexes, inc_dataset):
        return self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self)

    def extract(self, x):
        raw_features, features = self.convnet(x)
        if self.extract_no_act:
            return raw_features
        return features

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
        #features_normalized = F.normalize(features, p=2, dim=1)
        #weights_normalized = F.normalize(self.weights, p=2, dim=1)
#
        #return F.linear(features_normalized, weights_normalized)
        features_norm = features / (features.norm(dim=1)[:, None] + 1e-8)
        weights_norm = self.weights / (self.weights.norm(dim=1)[:, None] + 1e-8)
        similarities = torch.mm(features_norm, weights_norm.transpose(0, 1))
        return similarities

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(self.n_classes + n_classes, self.features_dim))

        #stdv = 1. / math.sqrt(self.features_dim)
        #new_weights.data.uniform_(-stdv, stdv)
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        if self.weights is not None:
            new_weights.data[:self.n_classes] = copy.deepcopy(self.weights.data)

        del self.weights
        self.weights = new_weights
        self.to(self.device)
        self.n_classes += n_classes

        return self

    def add_imprinted_classes(self, class_indexes, inc_dataset, network):
        # We are assuming the class indexes are contiguous!
        n_classes = self.n_classes
        self.add_classes(len(class_indexes))
        if n_classes == 0:
            return

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            new_weights.append(
                F.normalize(class_embeddings, p=2, dim=0) * avg_weights_norm
            )

        new_weights = torch.stack(new_weights)
        self.weights.data[-len(class_indexes):] = new_weights.to(self.device)
        return self


class LearnedScaler(nn.Module):
    def __init__(self, initial_value=1.):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(initial_value))

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
