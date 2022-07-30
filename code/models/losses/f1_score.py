import torch.nn as nn
from consts import CUTOFF
import torch


class F1Score(nn.Module):
    def __init__(self, threshold=CUTOFF):
        super(F1Score, self).__init__()
        self.threshold = threshold

    def forward(self, y_hat, y):
        tp = torch.sum(y * y_hat.float(), 0)
        tn = torch.sum((1 - y) * (1 - y_hat).float(), 0)
        fp = torch.sum((1 - y) * y_hat.float(), 0)
        fn = torch.sum(y * (1 - y_hat).float(), 0)

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        f1 = 2 * p * r / (p + r + 1e-7)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        f1.requires_grad = True

        return f1
