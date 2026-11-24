"""BaseTrainer: Unified training loop for BioCellAgents models.

Eliminates ~200 lines of duplicated training/validation/optimizer code
between train.py and train_complete.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import (
    amp_autocast,
    create_grad_scaler,
    get_device,
    load_checkpoint_safe,
)


class BaseTrainer:
    """Base class for training BioCellAgents models.

    Provides:
    - Differential learning rate optimizer (backbone vs head)
    - Training epoch with AMP support
    - Validation epoch with Dice/accuracy metrics
    - Checkpoint saving with best-model tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        backbone_lr: float = 1e-5,
        head_lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        output_dir: str = "checkpoints",
        task_name: str = "model",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.task_name = task_name

        self.device = get_device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda" and torch.cuda.is_available())
        self.model.to(self.device)

        self.optimizer = self._build_optimizer(backbone_lr, head_lr, weight_decay)
        self.scaler = create_grad_scaler(self.device, enabled=self.use_amp)

        self.best_metric = 0.0
        self.global_step = 0
        self._num_classes: Optional[int] = None

    def _build_optimizer(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with differential learning rates for backbone and head."""
        decay_backbone: List[nn.Parameter] = []
        no_decay_backbone: List[nn.Parameter] = []
        decay_head: List[nn.Parameter] = []
        no_decay_head: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = "backbone" in name
            is_no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name

            if is_backbone:
                (no_decay_backbone if is_no_decay else decay_backbone).append(param)
            else:
                (no_decay_head if is_no_decay else decay_head).append(param)

        param_groups = [
            {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": decay_head, "lr": head_lr, "weight_decay": weight_decay},
            {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups)

    def train_epoch(self, description: str = "Training") -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        non_blocking = self.device.type == "cuda"

        pbar = tqdm(self.train_loader, desc=description, leave=False)
        for batch in pbar:
            images = batch[0].to(self.device, non_blocking=non_blocking)
            targets = batch[1].to(self.device, non_blocking=non_blocking)
            self.optimizer.zero_grad()

            with amp_autocast(self.device, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.global_step += 1
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix(loss=f"{running_loss / max(total_samples, 1):.4f}")

        return running_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate_epoch(
        self, num_classes: Optional[int] = None, description: str = "Validating"
    ) -> Dict[str, float]:
        """Run validation epoch. Returns dict with 'loss', 'accuracy', and optionally 'dice'.

        The Dice metric is the **per-batch mean Dice**: for each batch, sum the
        per-class Dice over classes present in that batch, divide by ``num_classes``,
        then average over all batches. This matches ``evaluate.py`` exactly so that
        the training log and the standalone evaluation report the same number (the
        previous global/pooled Dice was more optimistic and disagreed with
        ``evaluate.py``).
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        total_elements = 0
        correct = 0
        non_blocking = self.device.type == "cuda"

        total_batch_dice = 0.0
        total_batches = 0

        for batch in tqdm(self.val_loader, desc=description, leave=False):
            images = batch[0].to(self.device, non_blocking=non_blocking)
            targets = batch[1].to(self.device, non_blocking=non_blocking)
            with amp_autocast(self.device, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total_elements += targets.numel()

            if num_classes:
                batch_dice = 0.0
                for c in range(num_classes):
                    pred_c = preds == c
                    target_c = targets == c
                    inter = (pred_c & target_c).sum().float()
                    union = pred_c.sum().float() + target_c.sum().float()
                    if union > 0:
                        batch_dice += (2.0 * inter / (union + 1e-6)).item()
                total_batch_dice += batch_dice / num_classes
                total_batches += 1

        results = {
            "loss": running_loss / max(total_samples, 1),
            "accuracy": correct / max(total_elements, 1),
        }

        if num_classes:
            results["dice"] = total_batch_dice / max(total_batches, 1)

        return results

    def train(
        self, epochs: int, metric_key: str = "dice", patience: int = 5, min_delta: float = 1e-3
    ) -> Dict[str, Any]:
        """Run full training loop with best-model checkpointing and early stopping.

        Args:
            epochs: Maximum number of epochs to train.
            metric_key: Validation metric used for best-model tracking and early stopping.
            patience: Stop training if the metric does not improve by at least
                ``min_delta`` for this many consecutive epochs.
            min_delta: Minimum improvement in the metric to count as progress.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_metrics: List[float] = []
        history: Dict[str, Any] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_metric": val_metrics,
        }

        patience_counter = 0
        best_epoch = 0
        last_trained_epoch = 0
        early_stopped = False

        for epoch in range(1, epochs + 1):
            last_trained_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs} [{self.task_name}]")
            print(f"{'='*60}")

            train_loss = self.train_epoch(f"Train {epoch}/{epochs}")
            val_results = self.validate_epoch(
                num_classes=getattr(self, "_num_classes", None), description=f"Val {epoch}/{epochs}"
            )

            val_loss = val_results["loss"]
            val_metric = val_results.get(metric_key, val_results.get("accuracy", 0.0))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val {metric_key.upper()}: {val_metric:.4f}")

            if val_metric > self.best_metric + min_delta:
                self.best_metric = val_metric
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model saved (metric={val_metric:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    early_stopped = True
                    print(
                        f"\n  Early stopping triggered after {epoch} epochs "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        self.save_checkpoint(last_trained_epoch, is_best=False)
        history["best_metric"] = self.best_metric
        history["best_epoch"] = best_epoch
        history["early_stopped"] = early_stopped
        history["epochs_trained"] = last_trained_epoch
        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        """Save model checkpoint.

        Best checkpoints are named ``best_{task_name}_model.pt`` (e.g.
        ``best_segmentation_model.pt``) to match the inference pipeline's
        default checkpoint path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if is_best:
            filename = f"best_{self.task_name}_model.pt"
        else:
            filename = f"{self.task_name}_epoch_{epoch}.pt"
        path = self.output_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint safely (supports full trainer dicts and raw state dicts)."""
        checkpoint = load_checkpoint_safe(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_metric = float(checkpoint.get("best_metric", 0.0))
            self.global_step = int(checkpoint.get("global_step", 0))
        else:
            self.model.load_state_dict(checkpoint)
        return checkpoint
