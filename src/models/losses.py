"""Shared loss functions for BioCellAgents segmentation and classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 6, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Joint CrossEntropy and Multi-Class Dice Loss for Cell Mask Segmentation."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_long = targets.long()
        targets_one_hot = F.one_hot(targets_long, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets.long())
        loss_dice = self.dice_loss(logits, targets)
        return (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice)
