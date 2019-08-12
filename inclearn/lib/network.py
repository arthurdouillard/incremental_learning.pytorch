import copy

import torch
from torch import nn
from torch.nn import functional as F

from inclearn.lib import factory, utils


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        convnet_kwargs={},
        classifier_kwargs={},
        postprocessor_kwargs={},
        init="kaiming",
        classifier_type="fc",
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False,
        attention_hook=False
    ):
        super(BasicNet, self).__init__()

        if postprocessor_kwargs["type"] == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs["type"] == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        else:
            self.post_processor = None

        self.convnet = factory.get_convnet(convnet_type, **convnet_kwargs)

        if classifier_type == "fc":
            self.classifier = Classifier(self.convnet.out_dim, device=device, **classifier_kwargs)
        elif classifier_type == "cosine":
            self.classifier = CosineClassifier(
                self.convnet.out_dim, device=device, **classifier_kwargs
            )
        elif classifier_type is None or classifier_type == "none":
            self.classifier = lambda x: x
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_type))

        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.attention_hook = attention_hook
        self.device = device

        if self.extract_no_act:
            print("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            print("No ReLU will be applied on features before feeding the classifier.")

        self.to(self.device)

    def on_task_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_task_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_task_end()

    def forward(self, x):
        outputs = self.convnet(x, attention_hook=self.attention_hook)
        logits = self.classifier(outputs[0] if self.classifier_no_act else outputs[1])

        if self.return_features:
            to_return = []
            if self.extract_no_act:
                to_return.append(outputs[0])
            else:
                to_return.append(outputs[1])

            to_return.append(logits)
            if self.attention_hook:
                to_return.append(outputs[2])

            return to_return
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

    def __init__(self, features_dim, *, use_bias, use_multi_fc, init, device, **kwargs):
        super().__init__()

        self.features_dim = features_dim
        self.use_bias = use_bias
        self.use_multi_fc = use_multi_fc
        self.init = init
        self.device = device

        self.n_classes = 0

        self.classifier = None

    def on_task_end(self):
        pass

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

    def __init__(self, features_dim, device, *, use_bias=False, proxy_per_class=1, **kwargs):
        super().__init__()

        self.n_classes = 0
        self.weights = None
        self.bias = None
        self.use_bias = use_bias
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device

    def on_task_end(self):
        pass

    def forward(self, features):
        features_norm = features / (features.norm(dim=1)[:, None] + 1e-8)
        weights_norm = self.weights / (self.weights.norm(dim=1)[:, None] + 1e-8)
        similarities = torch.mm(features_norm, weights_norm.transpose(0, 1))

        if self.use_bias:
            similarities = similarities + self.bias

        return similarities

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(
            torch.zeros(self.n_classes + self.proxy_per_class * n_classes, self.features_dim)
        )
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        if self.weights is not None:
            new_weights.data[:self.n_classes *
                             self.proxy_per_class] = copy.deepcopy(self.weights.data)

        del self.weights
        self.weights = new_weights

        if self.use_bias:
            new_bias = nn.Parameter(torch.zeros(self.n_classes + self.proxy_per_class * n_classes))
            nn.init.constant_(new_bias, 0.1)
            if self.bias is not None:
                new_bias.data[:self.n_classes *
                              self.proxy_per_class] = copy.deepcopy(self.bias.data)

            del self.bias
            self.bias = new_bias

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

            imprinted_weights = F.normalize(class_embeddings, p=2, dim=0) * avg_weights_norm

            for _ in range(self.proxy_per_class):
                # TODO could lead to bad performance if proxy_per_class > 1.
                # TODO add noise to differentiate them
                new_weights.append(imprinted_weights)

        new_weights = torch.stack(new_weights)
        self.weights.data[-len(class_indexes):] = new_weights.to(self.device)

        return self


class FactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwargs):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def forward(self, inputs):
        return self.scale * inputs


class HeatedUpScalar(nn.Module):
    def __init__(self, first_value, last_value, nb_steps, **kwargs):
        super().__init__()

        self.first_value = first_value
        self.step = (max(first_value, last_value) - min(first_value, last_value)) / nb_steps

        if first_value > last_value:
            self._factor = -1
        else:
            self._factor = 1

        self._increment = 0

        print("Heated-up factor is {}.".format(self.factor))

    def on_task_end(self):
        self._increment += 1
        print("Heated-up factor is {}.".format(self.factor))

    @property
    def factor(self):
        return self.first_value + (self._factor * self._increment * self.step)

    def forward(self, inputs):
        return self.factor * inputs



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
