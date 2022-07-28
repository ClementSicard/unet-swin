import torch.nn as nn
from .dice_loss import BinaryDiceLoss


class MixedLoss(nn.Module):
    def __init__(
        self,
        weights=[0.4, 0.6],
        smooth: float = 1.0,
        reduction: str = "none",
        p: int = 2,
    ):
        super(MixedLoss, self).__init__()
        self.weights = weights
        self.dice_loss = BinaryDiceLoss(
            smooth=smooth,
            p=p,
            reduction=reduction,
        )

        self.bce_loss = nn.modules.BCELoss()

    def forward(self, y_hat, y):
        bce_loss = self.bce_loss(y_hat, y)
        dice_loss = self.dice_loss(y_hat, y)

        return self.weights[0] * bce_loss + self.weights[1] * dice_loss
