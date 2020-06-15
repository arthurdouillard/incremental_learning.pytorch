import torch
from torch import optim
from torch.nn import functional as F

from inclearn.lib.network import (CalibrationWrapper, LinearModel, TemperatureScaling)


def calibrate(network, loader, device, indexes, calibration_type="linear"):
    """Corrects the bias for new classes.

    :param network: The logits extractor model, usually convnet+FC w/o final act.
    :param loader: The validation data loader.
    :param device: Device on which apply the computation.
    :param indexes: A list of tuple made a starting and ending indexes. They delimit
                    on which range of targets to apply the calibration. If given
                    several tuples, different models will be used per range.
    :return: A wrapper `CalibrationWrapper`.
    """
    logits, labels = _extract_data(network, loader, device)
    calibration_wrapper = _get_calibration_model(indexes, calibration_type).to(device)

    def eval():
        corrected_logits = calibration_wrapper(logits)
        loss = F.cross_entropy(corrected_logits, labels)
        loss.backward()
        return loss

    optimizer = optim.LBFGS(calibration_wrapper.parameters(), lr=0.01, max_iter=50)
    optimizer.step(eval)

    return calibration_wrapper


def _get_calibration_model(indexes, calibration_type):
    calibration_wrapper = CalibrationWrapper()

    for start_index, end_index in indexes:
        if calibration_type == "linear":
            model = LinearModel(alpha=1., beta=0.)
        elif calibration_type == "temperature":
            model = TemperatureScaling(temperature=1.)
        else:
            raise ValueError("Unknown calibration model {}.".format(calibration_type))

        calibration_wrapper.add_model(model, start_index, end_index)

    return calibration_wrapper


def _extract_data(network, loader, device):
    logits = []
    labels = []

    with torch.no_grad():
        for input_dict in loader:
            logits.append(network(input_dict["inputs"].to(device))["logits"])
            labels.append(input_dict["targets"].to(device))

        logits = torch.cat(logits).to(device)
        labels = torch.cat(labels).to(device)

    return logits, labels
