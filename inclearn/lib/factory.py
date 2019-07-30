import torch
from torch import optim

from inclearn import models
from inclearn.convnet import densenet, my_resnet, resnet, ucir_resnet
from inclearn.lib import data, samplers


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
    elif convnet_type == "resnet32":
        return resnet.resnet32(**kwargs)
    elif convnet_type == "rebuffi":
        return my_resnet.resnet_rebuffi(**kwargs)
    elif convnet_type == "densenet121":
        return densenet.densenet121(**kwargs)
    elif convnet_type == "ucir":
        return ucir_resnet.resnet32(**kwargs)

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
    elif args["model"] == "bic":
        return models.BiC(args)
    elif args["model"] == "icarlmixup":
        return models.ICarlMixUp(args)
    elif args["model"] == "ucir":
        return models.UCIR(args)
    elif args["model"] == "test":
        return models.Test(args)
    elif args["model"] == "ucir_test":
        return models.UCIRTest(args)

    raise NotImplementedError("Unknown model {}.".format(args["model"]))


def get_data(args):
    return data.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"],
        onehot=args["onehot"],
        increment=args["increment"],
        initial_increment=args["initial_increment"],
        sampler=get_sampler(args),
        data_path=args["data_path"]
    )


def set_device(args):
    device_type = args["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    args["device"] = device


def get_sampler(args):
    if args["sampler"] is None:
        return None

    sampler_type = args["sampler"].lower().strip()

    if sampler_type == "npair":
        return samplers.NPairSampler

    raise ValueError("Unknown sampler {}.".format(sampler_type))
