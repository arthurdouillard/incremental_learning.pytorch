import copy
import logging

from torch import nn

from inclearn.lib import factory

from .classifiers import Classifier, CosineClassifier, ProxyNCA, SoftTriple
from .postprocessors import ConstantScalar, FactorScalar, HeatedUpScalar

logger = logging.getLogger(__name__)


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        convnet_kwargs={},
        classifier_kwargs={},
        postprocessor_kwargs={},
        init="kaiming",
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False,
        attention_hook=False,
        rotations_predictor=False,
        dropout=0.
    ):
        super(BasicNet, self).__init__()

        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "constant":
            self.post_processor = ConstantScalar(**postprocessor_kwargs)
        else:
            self.post_processor = None
        logger.info("Post processor is: {}".format(self.post_processor))

        self.convnet = factory.get_convnet(convnet_type, **convnet_kwargs)

        if "type" not in classifier_kwargs:
            raise ValueError("Specify a classifier!", classifier_kwargs)
        if classifier_kwargs["type"] == "fc":
            self.classifier = Classifier(self.convnet.out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] == "cosine":
            self.classifier = CosineClassifier(
                self.convnet.out_dim, device=device, **classifier_kwargs
            )
        elif classifier_kwargs["type"] == "proxynca":
            self.classifier = ProxyNCA(self.convnet.out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] == "softtriple":
            self.classifier = SoftTriple(self.convnet.out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] is None or classifier_kwargs["type"] == "none":
            self.classifier = ConstantScalar()
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        if rotations_predictor:
            print("Using a rotations predictor.")
            self.rotations_predictor = nn.Linear(self.convnet.out_dim, 4)
        else:
            self.rotations_predictor = None

        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = ConstantScalar()

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

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_epoch_end()

    def forward(self, x):
        outputs = self.convnet(x, attention_hook=self.attention_hook)
        selected_outputs = outputs[0] if self.classifier_no_act else outputs[1]
        logits = self.classifier(self.dropout(selected_outputs))

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
        if hasattr(self.classifier, "add_classes"):
            self.classifier.add_classes(n_classes)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def add_custom_weights(self, weights):
        if hasattr(self.classifier, "add_custom_weights"):
            self.classifier.add_custom_weights(weights)

    def extract(self, x):
        raw_features, features = self.convnet(x)
        if self.extract_no_act:
            return raw_features
        return features

    def predict_rotations(self, inputs):
        if self.rotations_predictor is None:
            raise ValueError("Enable the rotations predictor.")
        return self.rotations_predictor(self.convnet(inputs, attention_hook=self.attention_hook)[1])

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "convnet":
            model = self.convnet
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    @property
    def group_parameters(self):
        groups = {"convnet": self.convnet.parameters()}

        if isinstance(self.classifier, nn.Module):
            groups["classifier"] = self.classifier.parameters()
        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.parameters()

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes
