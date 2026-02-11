import torch
import os
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_iter=-1):
        self.warmup_iters = warmup_iters
        self.current_iter = 0 if last_iter == -1 else last_iter
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            # Linear warmup phase
            return [
                base_lr * (self.current_iter + 1) / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            # Maintain the base learning rate after warmup
            return [base_lr for base_lr in self.base_lrs]

    def step(self, it=None):
        if it is None:
            it = self.current_iter + 1
        self.current_iter = it
        super(LinearWarmupScheduler, self).step(it)


