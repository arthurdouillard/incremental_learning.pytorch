from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Source:
        * https://github.com/weiaicunzai/pytorch-cifar100

    :param optimizer: SGD, Adam, etc.
    :param total_iters: number of iterations of the warmup phase. Usually number
                        of warmup epochs times the number of batches per epoch.
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
