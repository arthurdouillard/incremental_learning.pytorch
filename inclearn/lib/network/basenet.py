import copy
import logging

import torch
from torch import nn

from inclearn.lib import factory

from .classifiers import (Classifier, CosineClassifier, DomainClassifier, MCCosineClassifier)
from .postprocessors import FactorScalar, HeatedUpScalar, InvertedFactorScalar
from .word import Word2vec

logger = logging.getLogger(__name__)


class BasicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        convnet_kwargs={},
        classifier_kwargs={},
        postprocessor_kwargs={},
        wordembeddings_kwargs={},
        init="kaiming",
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False,
        attention_hook=False,
        rotations_predictor=False,
        gradcam_hook=False
    ):
        super(BasicNet, self).__init__()

        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "inverted_learned_scaling":
            self.post_processor = InvertedFactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
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
        elif classifier_kwargs["type"] == "mcdropout_cosine":
            self.classifier = MCCosineClassifier(
                self.convnet.out_dim, device=device, **classifier_kwargs
            )
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        if rotations_predictor:
            print("Using a rotations predictor.")
            self.rotations_predictor = nn.Linear(self.convnet.out_dim, 4)
        else:
            self.rotations_predictor = None

        if wordembeddings_kwargs:
            self.word_embeddings = Word2vec(**wordembeddings_kwargs, device=device)
        else:
            self.word_embeddings = None

        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.attention_hook = attention_hook
        self.gradcam_hook = gradcam_hook
        self.device = device

        self.domain_classifier = None

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

    def forward(
        self, x, rotation=False, index=None, features_processing=None, additional_features=None
    ):
        if hasattr(self,
                   "word_embeddings") and self.word_embeddings is not None and isinstance(x, list):
            words = x[1]
            x = x[0]
        else:
            words = None

        outputs = self.convnet(x)
        if words is not None:  # ugly to change
            outputs["word_embeddings"] = self.word_embeddings(words)

        if hasattr(self, "classifier_no_act") and self.classifier_no_act:
            selected_features = outputs["raw_features"]
        else:
            selected_features = outputs["features"]

        if features_processing is not None:
            selected_features = features_processing.fit_transform(selected_features)

        if rotation:
            outputs["rotations"] = self.rotations_predictor(outputs["features"])
            nb_inputs = len(x) // 4
            #for k in outputs.keys():
            #    if k != "rotations":
            #        if isinstance(outputs[k], list):
            #            outputs[k] = [elt[:32] for elt in outputs[k]]
            #        else:
            #            outputs[k] = outputs[k][:32]
        else:
            if additional_features is not None:
                clf_outputs = self.classifier(
                    torch.cat((selected_features, additional_features), 0)
                )
            else:
                clf_outputs = self.classifier(selected_features)
            outputs.update(clf_outputs)

        if hasattr(self, "gradcam_hook") and self.gradcam_hook:
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
        self.classifier.add_classes(n_classes)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def add_custom_weights(self, weights, **kwargs):
        self.classifier.add_custom_weights(weights, **kwargs)

    def extract(self, x):
        outputs = self.convnet(x)
        if self.extract_no_act:
            return outputs["raw_features"]
        return outputs["features"]

    def predict_rotations(self, inputs):
        if self.rotations_predictor is None:
            raise ValueError("Enable the rotations predictor.")
        return self.rotations_predictor(self.convnet(inputs)["features"])

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
        if hasattr(self, "gradcam_hook") and self.gradcam_hook and model == "convnet":
            for param in self.convnet.last_conv.parameters():
                param.requires_grad = True

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    def get_group_parameters(self):
        groups = {"convnet": self.convnet.parameters()}

        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.parameters()
        if hasattr(self.classifier, "new_weights"):
            groups["new_weights"] = self.classifier.new_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        if self.rotations_predictor:
            groups["rotnet"] = self.rotations_predictor.parameters()
        if hasattr(self.convnet, "last_block"):
            groups["last_block"] = self.convnet.last_block.parameters()
        if hasattr(self.classifier, "_negative_weights"
                  ) and isinstance(self.classifier._negative_weights, nn.Parameter):
            groups["neg_weights"] = self.classifier._negative_weights
        if self.domain_classifier is not None:
            groups["domain_clf"] = self.domain_classifier.parameters()

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

    def create_domain_classifier(self):
        self.domain_classifier = DomainClassifier(self.convnet.out_dim, device=self.device)
        return self.domain_classifier

    def del_domain_classifier(self):
        self.domain_classifier = None
