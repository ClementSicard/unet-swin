import torch.nn as nn
from .diff_patch_f1_loss import DifferentiablePatchF1Loss
from consts import CUTOFF, PATCH_SIZE


class MixedPatchF1Loss(nn.Module):
    def __init__(
        self,
        other_loss: nn.Module,
        threshold: float = CUTOFF,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        super(MixedPatchF1Loss, self).__init__()

        self.other_loss = other_loss
        self.threshold = threshold
        self.patch_size = patch_size
        self.diff_patch_f1_loss = DifferentiablePatchF1Loss(
            threshold=self.threshold,
            patch_size=self.patch_size,
        )

    def forward(self, y_hat, y):
        return 0.6 * self.other_loss(y_hat, y) + 0.4 * self.diff_patch_f1_loss(y_hat, y)
