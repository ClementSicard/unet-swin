# Copied from https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py, all rights reserved
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: sum{x^p} + sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth: float = 1.0, p: int = 2, reduction: str = "none"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"

        predict = torch.sigmoid(predict)
        predict = torch.flatten(predict)
        target = torch.flatten(target)

        intersection = (predict * target).sum()
        A_sum = torch.sum(predict)
        B_sum = torch.sum(target)

        loss = 1 - ((2.0 * intersection + self.smooth) / (A_sum + B_sum + self.smooth))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))
