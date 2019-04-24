import argparse


def get_parser():
    # yapf: disable

    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer.")

    # Model related:
    parser.add_argument("-m", "--model", default="icarl", type=str,
                        help="Incremental learner to train.")
    parser.add_argument("-c", "--convnet", default="resnet18", type=str,
                        help="Backbone convnet.")
    parser.add_argument("--dropout", default=0., type=float,
                        help="Dropout value.")
    parser.add_argument("-he", "--herding", default=None, type=str,
                        help="Method to gather previous tasks' examples.")
    parser.add_argument("-memory", "--memory-size", default=2000, type=int,
                        help="Max number of storable examplars.")
    parser.add_argument("-temp", "--temperature", default=1, type=int,
                        help="Temperature used to soften the predictions.")

    # Data related:
    parser.add_argument("-d", "--dataset", default="iCIFAR100", type=str,
                        help="Dataset to test on.")
    parser.add_argument("-inc", "--increment", default=10, type=int,
                        help="Number of class to add per task.")
    parser.add_argument("-b", "--batch-size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-w", "--workers", default=10, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-v", "--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-random", "--random-classes", action="store_true", default=False,
                        help="Randomize classes order of increment")
    parser.add_argument("-max-task", "--max-task", default=None, type=int,
                        help="Cap the number of tasks.")

    # Training related:
    parser.add_argument("-lr", "--lr", default=2., type=float,
                        help="Learning rate.")
    parser.add_argument("-wd", "--weight-decay", default=0.00001, type=float,
                        help="Weight decay.")
    parser.add_argument("-sc", "--scheduling", default=[50, 64], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-lr-decay", "--lr-decay", default=1/5, type=int,
                        help="LR multiplied by it.")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                        help="Optimizer to use.")
    parser.add_argument("-e", "--epochs", default=70, type=int,
                        help="Number of epochs per task.")

    # Misc:
    parser.add_argument("--device", default=0, type=int,
                        help="GPU index to use, for cpu use -1.")
    parser.add_argument("--name", default="exp",
                        help="Experience name")
    parser.add_argument("-seed", "--seed", default=1, type=int,
                        help="Random seed.")

    return parser
