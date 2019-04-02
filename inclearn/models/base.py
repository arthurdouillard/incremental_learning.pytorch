import abc
import logging

from torch import nn


LOGGER = logging.Logger("IncLearn", level="INFO")


class IncrementalLearner(abc.ABC, nn.Module):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self, *args, **kwargs)

    def set_task_info(self, task, total_n_classes, increment, n_train_data,
                      n_test_data):
        self._task = task
        self._task_size = increment
        self._total_n_classes = total_n_classes
        self._n_train_data = n_train_data
        self._n_test_data = n_test_data

    def before_task(self, data_loader):
        LOGGER.info("Before task")
        self.eval()
        self._before_task(data_loader)

    def train_task(self, data_loader):
        LOGGER.info("train task")
        self.train()
        self._train_task(data_loader)

    def after_task(self, data_loader):
        LOGGER.info("after task")
        self.eval()
        self._after_task(data_loader)

    def eval_task(self, data_loader):
        LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory_indexes(self):
        return []

    def _before_task(self, data_loader):
        pass

    def _train_task(self, data_loader):
        raise NotImplementedError

    def _after_task(self, data_loader):
        pass

    @property
    def _new_task_index(self):
        return self._task * self._task_size

