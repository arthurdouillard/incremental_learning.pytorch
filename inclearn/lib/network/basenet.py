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
        gradcam_hook=False,
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
        self.gradcam_hook = gradcam_hook
        self.device = device

        if self.gradcam_hook:
            self._hooks = [None, None]
            logger.info("Setting gradcam hook for gradients + activations of last conv.")
            self.set_gradcam_hook()
        if self.extract_no_act:
            logger.info("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            logger.info("No ReLU will be applied on features before feeding the classifier.")

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
        outputs = self.convnet(x)

        if self.classifier_no_act:
            selected_features = outputs["raw_features"]
        else:
            selected_features = outputs["features"]
        logits = self.classifier(self.dropout(selected_features))

        outputs["logits"] = logits

        if self.gradcam_hook:
            outputs["gradcam_gradients"] = self._gradcam_gradients
            outputs["gradcam_activations"] = self._gradcam_activations

        return outputs

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
        outputs = self.convnet(x)
        if self.extract_no_act:
            return outputs["raw_features"]
        return outputs["features"]

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
        if self.gradcam_hook and model == "convnet":
            for param in self.convnet.last_conv.parameters():
                param.requires_grad = True

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

    def unset_gradcam_hook(self):
        self._hooks[0].remove()
        self._hooks[1].remove()
        self._hooks[0] = None
        self._hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)
