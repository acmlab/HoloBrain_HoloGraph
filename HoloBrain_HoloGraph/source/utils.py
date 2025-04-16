import torch
import os
import logging
from torch.optim.lr_scheduler import _LRScheduler

def logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        current_iter = max(self.last_epoch, 0)
        if current_iter < self.warmup_iters:
            return [base_lr * (current_iter + 1) / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return self.base_lrs

        
def compute_weighted_metrics(preds, gts):
    num_classes = len(torch.unique(gts))
    device = preds.device

    class_counts = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        class_counts[cls] = (gts == cls).sum()

    correct = (preds == gts).sum().item()
    acc = correct / gts.size(0)

    precision_list, recall_list, f1_list = [], [], []
    for cls in range(num_classes):
        tp = ((preds == cls) & (gts == cls)).sum().item()
        fp = ((preds == cls) & (gts != cls)).sum().item()
        fn = ((preds != cls) & (gts == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        weight = class_counts[cls] / class_counts.sum()
        precision_list.append(precision * weight)
        recall_list.append(recall * weight)
        f1_list.append(f1 * weight)

    pre = sum(precision_list).item()
    recall = sum(recall_list).item()
    f1 = sum(f1_list).item()

    return acc, pre, recall, f1
