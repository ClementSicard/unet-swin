import torch
import torch.nn as nn
from models.losses.f1_score import F1Score
from consts import PATCH_SIZE, CUTOFF
import numpy as np


class DifferentiablePatchF1Loss(nn.Module):
    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        threshold: float = CUTOFF,
    ) -> None:
        super(DifferentiablePatchF1Loss, self).__init__()

        self.patch_size = patch_size
        self.threshold = threshold
        self.f1_score = F1Score(threshold=threshold)

    def forward(self, y_hat, y):
        w = y_hat.shape[-1]
        size = self.patch_size if w > 250 else self.patch_size // 2
        # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
        h_patches = y.shape[-2] // size
        w_patches = y.shape[-1] // size

        tmp_patches_hat = y_hat.reshape(-1, 1, h_patches, size, w_patches, size).mean(
            (-1, -3)
        )
        tmp_patches = y.reshape(-1, 1, h_patches, size, w_patches, size).mean((-1, -3))

        patches_hat = torch.where(
            tmp_patches_hat >= self.threshold,
            torch.ones_like(tmp_patches_hat),
            torch.zeros_like(tmp_patches_hat),
        )
        patches = torch.where(
            tmp_patches >= self.threshold,
            torch.ones_like(tmp_patches),
            torch.zeros_like(tmp_patches),
        )

        patch_f1 = self.f1_score(y=patches, y_hat=patches_hat)

        return 1 - torch.mean(patch_f1)
