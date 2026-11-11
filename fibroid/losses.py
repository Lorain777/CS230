import torch
import torch.nn.functional as F
from torch import nn


class DiceBCELoss(nn.Module):
    """Binary cross-entropy plus soft Dice loss, both computed from logits.

    Dice is taken over the whole batch rather than per slice, so slices with
    no fibroid do not produce a degenerate 0/0 term.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum()
        dice = (2 * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        return bce + 1 - dice
