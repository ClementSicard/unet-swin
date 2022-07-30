import torch.nn as nn
from .diff_f1_loss import DifferentiableF1Loss
from consts import CUTOFF


class MixedF1Loss(nn.Module):
    def __init__(
        self,
        other_loss: nn.Module,
        threshold: float = CUTOFF,
        f1_loss_weight: float = 0.4,
    ) -> None:
        super(MixedF1Loss, self).__init__()

        self.other_loss = other_loss
        self.threshold = threshold
        self.diff_f1_loss = DifferentiableF1Loss(threshold=self.threshold)
        self.f1_loss_weight = f1_loss_weight

    def forward(self, y_hat, y):
        w = self.f1_loss_weight

        return (1 - w) * self.other_loss(y_hat, y) + w * self.diff_f1_loss(y_hat, y)
