import torch.nn as nn
from .diff_f1_loss import DifferentiableF1Loss
from consts import CUTOFF


class MixedF1Loss(nn.Module):
    def __init__(
        self,
        other_loss: nn.Module,
        threshold: float = CUTOFF,
    ) -> None:
        super(MixedF1Loss, self).__init__()

        self.other_loss = other_loss
        self.threshold = threshold
        self.diff_f1_loss = DifferentiableF1Loss(threshold=self.threshold)

    def forward(self, y_hat, y):
        return 0.6 * self.other_loss(y_hat, y) + 0.4 * self.diff_f1_loss(y_hat, y)
