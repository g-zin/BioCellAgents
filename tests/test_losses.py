"""Unit tests for BioCellAgents loss functions."""
import pytest
import torch
import torch.nn as nn


class TestDiceLoss:
    """Tests for DiceLoss."""

    def test_perfect_prediction_zero_loss(self):
        """Dice loss should be ~0 when prediction perfectly matches target."""
        from src.models.losses import DiceLoss
        loss_fn = DiceLoss(num_classes=3)
        # Targets are 3D: (B, H, W) - include all classes so every class has pixels
        targets = torch.tensor([[[0, 1], [1, 2]]])  # (1, 2, 2)
        # Logits: (B, C, H, W) - strongly favor correct class at each position
        logits = torch.zeros(1, 3, 2, 2)
        logits[0, 0, 0, 0] = 10.0  # class 0 at (0,0)
        logits[0, 1, 0, 1] = 10.0  # class 1 at (0,1)
        logits[0, 1, 1, 0] = 10.0  # class 1 at (1,0)
        logits[0, 2, 1, 1] = 10.0  # class 2 at (1,1)
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_prediction_high_loss(self):
        """Dice loss should be high when prediction is completely wrong."""
        from src.models.losses import DiceLoss
        loss_fn = DiceLoss(num_classes=2)
        targets = torch.ones(1, 2, 2, dtype=torch.long)  # All class 1
        logits = torch.zeros(1, 2, 2, 2)
        logits[0, 0] = 10.0  # Strongly predict class 0
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.5

    def test_output_range(self):
        """Dice loss should be in [0, 1] range."""
        from src.models.losses import DiceLoss
        loss_fn = DiceLoss(num_classes=4)
        torch.manual_seed(42)
        logits = torch.randn(4, 4, 8, 8)
        targets = torch.randint(0, 4, (4, 8, 8))
        loss = loss_fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_different_num_classes(self):
        """DiceLoss should work with different numbers of classes."""
        from src.models.losses import DiceLoss
        for nc in [2, 3, 6, 10]:
            loss_fn = DiceLoss(num_classes=nc)
            logits = torch.randn(2, nc, 4, 4)
            targets = torch.randint(0, nc, (2, 4, 4))
            loss = loss_fn(logits, targets)
            assert loss.item() >= 0.0


class TestCombinedSegmentationLoss:
    """Tests for CombinedSegmentationLoss."""

    def test_forward_pass(self):
        """Combined loss should produce a valid scalar."""
        from src.models.losses import CombinedSegmentationLoss
        loss_fn = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0.0

    def test_ce_weight_zero(self):
        """With ce_weight=0, loss should equal dice loss."""
        from src.models.losses import CombinedSegmentationLoss
        loss_fn = CombinedSegmentationLoss(ce_weight=0.0, dice_weight=1.0)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = loss_fn(logits, targets)
        dice_only = loss_fn.dice_loss(logits, targets)
        assert abs(loss.item() - dice_only.item()) < 1e-5

    def test_dice_weight_zero(self):
        """With dice_weight=0, loss should equal CE loss."""
        from src.models.losses import CombinedSegmentationLoss
        loss_fn = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=0.0)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = loss_fn(logits, targets)
        ce_only = nn.CrossEntropyLoss()(logits, targets)
        assert abs(loss.item() - ce_only.item()) < 1e-5

    def test_class_weights(self):
        """Loss should accept class weights."""
        from src.models.losses import CombinedSegmentationLoss
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = CombinedSegmentationLoss(class_weights=weights)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_gradient_flow(self):
        """Gradients should flow through the combined loss."""
        from src.models.losses import CombinedSegmentationLoss
        loss_fn = CombinedSegmentationLoss()
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0