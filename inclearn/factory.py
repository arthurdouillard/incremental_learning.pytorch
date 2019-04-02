import torch
from torch import optim

import resnet
import models
import data


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError


def get_resnet(resnet_type):
    if resnet_type == "resnet18":
        return resnet.resnet18()
    elif resnet_type == "resnet34":
        return resnet.resnet101()

    raise NotImplementedError(resnet_type)


def get_model(args):
    if args.model == "icarl":
        return models.ICarl(args)

    raise NotImplementedError(args.model)


def get_data(args, train=True):
    if args.dataset.lower() in ("icifar100", "cifar100"):
        return data.iCIFAR100(increment=args.increment, train=train)
    if args.dataset.lower() in ("icifar10", "cifar10"):
        return data.iCIFAR10(increment=args.increment, train=train)

    raise NotImplementedError(args.dataset)


def get_loader(dataset, args):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )


def set_device(args):
    device_type = args.device

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    args.device = device
