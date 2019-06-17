import torch
from torch import optim

from inclearn import models
from inclearn.convnet import densenet, my_resnet, resnet
from inclearn.lib import data


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

    raise NotImplementedError


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet34":
        return resnet.resnet34(**kwargs)
    elif convnet_type == "rebuffi":
        return my_resnet.resnet_rebuffi()
    elif convnet_type == "densenet121":
        return densenet.densenet121(**kwargs)

    raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(args):
    if args["model"] == "icarl":
        return models.ICarl(args)
    elif args["model"] == "lwf":
        return models.LwF(args)
    elif args["model"] == "e2e":
        return models.End2End(args)
    elif args["model"] == "medic":
        return models.Medic(args)
    elif args["model"] == "focusforget":
        return models.FocusForget(args)
    elif args["model"] == "fixed":
        return models.FixedRepresentation(args)

    raise NotImplementedError(args["model"])


def get_data(args):
    return data.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"]
    )


def set_device(args):
    device_type = args["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    args["device"] = device
