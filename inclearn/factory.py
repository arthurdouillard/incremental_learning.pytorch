import torch
from torch import optim

from inclearn import data, models, resnet


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

    raise NotImplementedError


def get_resnet(resnet_type, **kwargs):
    if resnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif resnet_type == "resnet34":
        return resnet.resnet101(**kwargs)

    raise NotImplementedError(resnet_type)


def get_model(args):
    if args["model"] == "icarl":
        return models.ICarl(args)

    raise NotImplementedError(arg["model"])


def get_data(args, train=True, classes_order=None):
    dataset_name = args["dataset"].lower()

    if dataset_name in ("icifar100", "cifar100"):
        dataset = data.iCIFAR100
    elif dataset_name in ("icifar10", "cifar10"):
        dataset = data.iCIFAR10
    else:
        raise NotImplementedError(dataset_name)

    return dataset(
        increment=args["increment"],
        train=train,
        randomize_class=args["random_classes"],
        classes_order=classes_order
    )

def set_device(args):
    device_type = args["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    args["device"] = device
