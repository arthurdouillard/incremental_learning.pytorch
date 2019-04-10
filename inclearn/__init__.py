import random

import numpy as np
import torch

import data
import factory
import models
import resnet
import utils

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
