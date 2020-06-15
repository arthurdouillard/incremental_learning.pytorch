import torch
from torch.nn import functional as F


def unsupervised_rotations(inputs, memory_flags, network, apply_on="all", factor=1.0, **kwargs):
    """Rotates inputs by 90° four times, and predict the angles.

    References:
        * Spyros Gidaris, Praveer Singh, Nikos Komodakis
          Unsupervised Representation Learning by Predicting Image Rotations
          ICLR 2018

    :param inputs: Inputs images.
    :param memory_flags: Flags signaling if the inputs are from the memory.
    :param network: Network used to predict rotations.
    :param config: A dict of configuration for this loss.
    :return: A float scalar loss.
    """
    if apply_on == "all":
        selected_inputs = inputs
    elif apply_on == "old":
        selected_inputs = inputs[memory_flags.eq(1.)]
    elif apply_on == "new":
        selected_inputs = inputs[memory_flags.eq(0.)]
    else:
        raise ValueError("Invalid apply for rotation prediction: {}.".format(apply_on))

    if len(selected_inputs) == 0:
        return torch.tensor(0.)

    rotated_inputs = [selected_inputs]
    angles = [torch.zeros(len(selected_inputs))]

    for ang in range(1, 4):
        rotated_inputs.append(selected_inputs.rot90(ang, (2, 3)))
        angles.append(torch.ones(len(selected_inputs)) * ang)

    rotated_inputs = torch.cat(rotated_inputs)
    angles = torch.cat(angles).long().to(inputs.device)
    outputs = network(rotated_inputs, rotation=True, index=len(inputs))
    loss = factor * F.cross_entropy(outputs["rotations"], angles)

    return loss, outputs
