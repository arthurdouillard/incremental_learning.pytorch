import torch
import torch.nn as nn


def get_gradcam_hook(model):
    if isinstance(model, nn.DataParallel):
        gradients = [None for _ in model.device_ids]
        activations = [None for _ in model.device_ids]

        def backward_hook(module, grad_input, grad_output):
            gradients[model.device_ids.index(grad_output[0].device.index)] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            activations[model.device_ids.index(output.device.index)] = output
            return None
    else:
        gradients = [None]
        activations = [None]

        def backward_hook(module, grad_input, grad_output):
            gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            activations[0] = output
            return None

    return gradients, activations, backward_hook, forward_hook
