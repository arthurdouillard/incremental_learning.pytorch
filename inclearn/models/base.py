import abc
import logging

import torch

LOGGER = logging.Logger("IncLearn", level="INFO")

logger = logging.getLogger(__name__)


class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, *args, **kwargs):
        self._network = None

    def set_task_info(self, task, total_n_classes, increment, n_train_data, n_test_data, n_tasks):
        self._task = task
        self._task_size = increment
        self._total_n_classes = total_n_classes
        self._n_train_data = n_train_data
        self._n_test_data = n_test_data
        self._n_tasks = n_tasks

    def before_task(self, train_loader, val_loader):
        LOGGER.info("Before task")
        self.eval()
        self._before_task(train_loader, val_loader)

    def train_task(self, train_loader, val_loader):
        LOGGER.info("train task")
        self.train()
        self._train_task(train_loader, val_loader)

    def after_task(self, inc_dataset):
        LOGGER.info("after task")
        self.eval()
        self._after_task(inc_dataset)

    def eval_task(self, data_loader):
        LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        return None

    def get_val_memory(self):
        return None

    def _before_task(self, data_loader, val_loader):
        pass

    def _train_task(self, train_loader, val_loader):
        raise NotImplementedError

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def save_metadata(self, path):
        pass

    def load_metadata(self, path):
        pass

    @property
    def _new_task_index(self):
        return self._task * self._task_size

    @property
    def inc_dataset(self):
        return self.__inc_dataset

    @inc_dataset.setter
    def inc_dataset(self, inc_dataset):
        self.__inc_dataset = inc_dataset

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network_path):
        if self._network is not None:
            del self._network

        logger.info("Loading model from {}.".format(network_path))
        self._network = torch.load(network_path)
        self._network.to(self._device)
        self._network.device = self._device
        self._network.classifier.device = self._device

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
