import json
import argparse

import torch

from inclearn import factory

torch.manual_seed(1)


parser = argparse.ArgumentParser("IncLearner",
                                 description="Incremental Learning trainer.")

# Model related:
parser.add_argument("-m", "--model", default="icarl", type=str,
                    help="Incremental learner to train.")
parser.add_argument("-c", "--convnet", default="resnet18", type=str,
                    help="Backbone convnet.")
parser.add_argument("-he", "--herding", default=None, type=str,
                    help="Method to gather previous tasks' examples.")
parser.add_argument("-memory", "--memory-size", default=2000, type=int,
                    help="Max number of storable examplars.")

# Data related:
parser.add_argument("-d", "--dataset", default="iCIFAR100", type=str,
                    help="Dataset to test on.")
parser.add_argument("-inc", "--increment", default=10, type=int,
                    help="Number of class to add per task.")
parser.add_argument("-b", "--batch-size", default=128, type=int,
                    help="Batch size.")
parser.add_argument("-w", "--workers", default=10, type=int,
                    help="Number of workers preprocessing the data.")

# Training related:
parser.add_argument("-lr", "--lr", default=0.001, type=float,
                    help="Learning rate")
parser.add_argument("-sc", "--scheduling", default=[], nargs="*",
                    help="Epoch step where to reduce the learning rate.")
parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                    help="Optimizer to use.")
parser.add_argument("-e", "--epochs", default=70, type=int,
                    help="Number of epochs per task.")

# Misc:
parser.add_argument("--device", default=0, type=int,
                    help="GPU index to use, for cpu use -1.")


args = parser.parse_args()

factory.set_device(args)

train_set = factory.get_data(args, train=True)
test_set = factory.get_data(args, train=False)

train_loader = factory.get_loader(train_set, args)
test_loader = factory.get_loader(test_set, args)

model = factory.get_model(args)

for task in range(0, train_set.total_n_classes // args.increment):
    # Setting current task's classes:
    train_set.set_classes_range(high=(task + 1) * args.increment)
    test_set.set_classes_range(high=(task + 1) * args.increment)

    model.set_task_info(
        task,
        train_set.total_n_classes,
        args.increment,
        len(train_set),
        len(test_set)
    )

    model.before_task(train_loader)
    model.train_task(train_loader)
    model.after_task(train_loader)

    ypred, ytrue = model.eval_task(train_loader)