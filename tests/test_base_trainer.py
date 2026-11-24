"""Offline CPU unit tests for src.models.base_trainer.BaseTrainer."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_trainer import BaseTrainer


class _TinyClassifier(nn.Module):
    def __init__(self, in_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Linear(in_dim, 8)
        self.head = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))


class _ConstantSegModel(nn.Module):
    """Segmentation model with trainable parameter that predicts a fixed class everywhere."""

    def __init__(self, num_classes: int = 4, fixed_class: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.fixed_class = fixed_class
        self.backbone = nn.Conv2d(3, 4, kernel_size=1)
        self.head = nn.Conv2d(4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep gradient connection to parameters while emitting deterministic argmax
        base = self.head(self.backbone(x)) * 0.0
        offset = torch.full_like(base, -10.0)
        offset[:, self.fixed_class, :, :] = 10.0
        return base + offset


class _TinySegModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, kernel_size=1)
        self.head = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))


class TestBaseTrainer:
    """Behavior and regression tests for BaseTrainer."""

    def test_cpu_init_emits_no_grad_scaler_warnings(self, tmp_path: Path):
        x = torch.randn(4, 8)
        y = torch.randint(0, 3, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer = BaseTrainer(
                model=_TinyClassifier(),
                train_loader=loader,
                val_loader=loader,
                criterion=nn.CrossEntropyLoss(),
                use_amp=True,
                output_dir=str(tmp_path),
                task_name="clf",
                device="cpu",
            )

        assert trainer.device == torch.device("cpu")
        assert trainer.use_amp is False
        assert trainer.scaler is None
        assert caught == []

    def test_classification_validate_epoch_accuracy_in_unit_interval(self, tmp_path: Path):
        torch.manual_seed(0)
        x = torch.randn(10, 8)
        y = torch.randint(0, 3, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        trainer = BaseTrainer(
            model=_TinyClassifier(),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(tmp_path),
            task_name="clf",
            device="cpu",
        )
        metrics = trainer.validate_epoch()
        assert metrics["loss"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "dice" not in metrics

    def test_segmentation_validate_epoch_pixel_accuracy_and_dice_bug3_regression(
        self, tmp_path: Path
    ):
        """Regression for Bug #3: (B, H, W) targets must divide by B*H*W, not B."""
        images = torch.zeros(2, 3, 16, 16)
        masks = torch.zeros(2, 16, 16, dtype=torch.long)
        # Exactly 50% of pixels are class 0 and 50% are class 1
        masks[:, :8, :] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        # Model always predicts class 0 -> exact pixel accuracy must be 0.5 (never 256.0!)
        trainer = BaseTrainer(
            model=_ConstantSegModel(num_classes=4, fixed_class=0),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(tmp_path),
            task_name="seg",
            device="cpu",
        )
        metrics = trainer.validate_epoch(num_classes=4)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["accuracy"] == pytest.approx(0.5, abs=1e-6)
        assert "dice" in metrics
        # Class 0: inter=256, pred=512, target=256 -> dice = 2*256 / 768 = 2/3; others = 0
        # Per-batch mean Dice over 4 classes = (2/3) / 4 = 1/6
        assert metrics["dice"] == pytest.approx((2.0 / 3.0) / 4.0, abs=1e-4)

    def test_train_epoch_increments_global_step(self, tmp_path: Path):
        torch.manual_seed(1)
        images = torch.randn(6, 3, 8, 8)
        masks = torch.randint(0, 4, (6, 8, 8))
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        trainer = BaseTrainer(
            model=_TinySegModel(num_classes=4),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(tmp_path),
            task_name="seg",
            device="cpu",
        )
        assert trainer.global_step == 0
        loss1 = trainer.train_epoch()
        assert loss1 >= 0.0
        assert trainer.global_step == 3  # 6 samples / batch_size 2
        trainer.train_epoch()
        assert trainer.global_step == 6

    def test_save_and_load_checkpoint_roundtrip_and_raw_state_dict(self, tmp_path: Path):
        torch.manual_seed(2)
        x = torch.randn(4, 8)
        y = torch.randint(0, 3, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=2)

        trainer = BaseTrainer(
            model=_TinyClassifier(),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(tmp_path),
            task_name="clf",
            device="cpu",
        )
        trainer.train_epoch()
        trainer.best_metric = 0.88
        saved_weight = trainer.model.head.weight.detach().clone()

        ckpt_path = trainer.save_checkpoint(epoch=3, is_best=True)
        assert ckpt_path.exists()

        # Mutate state before restoring
        with torch.no_grad():
            trainer.model.head.weight.zero_()
        trainer.best_metric = 0.0
        trainer.global_step = 0

        loaded = trainer.load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 3
        assert trainer.best_metric == pytest.approx(0.88)
        assert trainer.global_step == 2
        assert torch.allclose(trainer.model.head.weight, saved_weight)

        # Also support raw state_dict files
        raw_path = tmp_path / "raw_state_dict.pt"
        torch.save(trainer.model.state_dict(), raw_path)
        with torch.no_grad():
            trainer.model.head.weight.zero_()
        trainer.load_checkpoint(raw_path)
        assert torch.allclose(trainer.model.head.weight, saved_weight)

    def test_save_checkpoint_creates_nonexistent_output_dir_before_train(self, tmp_path: Path):
        x = torch.randn(4, 8)
        y = torch.randint(0, 3, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=2)

        missing_dir = tmp_path / "never_created" / "nested_ckpts"
        assert not missing_dir.exists()

        trainer = BaseTrainer(
            model=_TinyClassifier(),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(missing_dir),
            task_name="clf",
            device="cpu",
        )
        ckpt_path = trainer.save_checkpoint(epoch=1, is_best=True)
        assert missing_dir.is_dir()
        assert ckpt_path.is_file()

    def test_train_early_stopping_and_checkpoint_artifacts(self, tmp_path: Path):
        images = torch.zeros(2, 3, 8, 8)
        masks = torch.zeros(2, 8, 8, dtype=torch.long)
        masks[:, :4, :] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        trainer = BaseTrainer(
            model=_ConstantSegModel(num_classes=4, fixed_class=0),
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            output_dir=str(tmp_path),
            task_name="seg",
            device="cpu",
        )
        trainer._num_classes = 4
        history = trainer.train(epochs=10, metric_key="dice", patience=2, min_delta=1e-3)

        assert history["early_stopped"] is True
        # Epoch 1 improves over 0.0 -> best_epoch=1; epochs 2 and 3 stall -> stops at epoch 3
        assert history["best_epoch"] == 1
        assert history["epochs_trained"] == 3
        assert (tmp_path / "best_seg_model.pt").exists()
        assert (tmp_path / "seg_epoch_3.pt").exists()
