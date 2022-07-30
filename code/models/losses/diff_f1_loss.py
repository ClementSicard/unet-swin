import torch
import torch.nn as nn
from .f1_score import F1Score
from consts import CUTOFF


class DifferentiableF1Loss(nn.Module):
    def __init__(self, threshold=CUTOFF):
        super(DifferentiableF1Loss, self).__init__()
        self.threshold = threshold
        self.f1_score = F1Score(threshold=threshold)

    def forward(self, y_hat, y):
        aug_y_hat = torch.where(
            y_hat >= self.threshold, torch.ones_like(y_hat), torch.zeros_like(y_hat)
        )
        aug_y = torch.where(
            y >= self.threshold, torch.ones_like(y), torch.zeros_like(y)
        )

        f1 = self.f1_score(y_hat=aug_y_hat, y=aug_y)

        return 1 - torch.mean(f1)
