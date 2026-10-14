"""Comprehensive evaluation module for BioCellAgents models.

Provides unified evaluation for all 3 model types:
- Cell Segmentation (Dice, IoU, per-class metrics)
- Cell Classification (Accuracy, F1, per-class P/R/F1)
- Organ Classification (Accuracy, F1, per-class P/R/F1)

Includes cross-validation support and detailed report generation.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate a classification model and return detailed metrics.

    Args:
        model: Classification model.
        dataloader: Validation data loader.
        device: Target device.
        class_names: List of class name strings.

    Returns:
        Dict with accuracy, macro_f1, per_class metrics, confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_recall_fscore_support, confusion_matrix,
    )

    model.eval()
    all_preds, all_targets = [], []

    for images, labels in tqdm(dataloader, desc="Evaluating Classification"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    labels_list = list(range(len(class_names)))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_list, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=labels_list)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(np.sum(np.array(all_targets) == i)),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_targets),
    }


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Evaluate a segmentation model and return Dice/IoU metrics.

    The headline ``dice`` is the **per-batch mean Dice** (average over batches of
    the per-batch class-mean Dice), consistent with ``evaluate.py`` and the
    trainer's ``validate_epoch`` so all reported numbers are directly comparable.
    ``per_class_dice`` / ``per_class_iou`` are the global per-class breakdown
    (pooled over the whole set) for diagnostics.

    Args:
        model: Segmentation model (expects image, returns logits).
        dataloader: Validation data loader (yields images, masks).
        device: Target device.
        num_classes: Number of segmentation classes.

    Returns:
        Dict with dice, iou, per_class_dice, per_class_iou, num_samples.
    """
    model.eval()
    all_preds, all_targets = [], []
    total_batch_dice = 0.0
    total_batches = 0

    for images, masks in tqdm(dataloader, desc="Evaluating Segmentation"):
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        masks = masks.cpu()

        # Per-batch mean Dice (canonical metric, matches evaluate.py)
        batch_dice = 0.0
        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (masks == c)
            inter = (pred_c & target_c).sum().float()
            union = pred_c.sum().float() + target_c.sum().float()
            if union > 0:
                batch_dice += (2.0 * inter / (union + 1e-6)).item()
        total_batch_dice += batch_dice / num_classes
        total_batches += 1

        all_preds.append(preds)
        all_targets.append(masks)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Global per-class Dice/IoU (diagnostic breakdown)
    per_class_dice = []
    per_class_iou = []
    for c in range(num_classes):
        pred_c = (all_preds == c).astype(np.float32)
        target_c = (all_targets == c).astype(np.float32)
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c)
        dice = 2 * intersection / (union + 1e-8)
        iou = intersection / (np.sum(pred_c + target_c - pred_c * target_c) + 1e-8)
        per_class_dice.append(round(float(dice), 4))
        per_class_iou.append(round(float(iou), 4))

    return {
        "dice": round(total_batch_dice / max(total_batches, 1), 4),
        "iou": round(float(np.mean(per_class_iou)), 4),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "num_samples": len(all_preds),
    }


def run_full_evaluation(
    model_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Run full evaluation based on model type.

    Args:
        model_type: One of "segmentation", "cell_classification", "organ_classification".
        model: The model to evaluate.
        dataloader: Validation data loader.
        device: Target device.
        class_names: Class names (required for classification).
        num_classes: Number of classes (for segmentation).

    Returns:
        Dict with all evaluation metrics.
    """
    if model_type == "segmentation":
        return evaluate_segmentation(model, dataloader, device, num_classes)
    elif model_type in ("cell_classification", "organ_classification"):
        if class_names is None:
            raise ValueError("class_names required for classification evaluation")
        return evaluate_classification(model, dataloader, device, class_names)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def generate_report(
    results: Dict[str, Any],
    model_name: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate a human-readable evaluation report.

    Args:
        results: Evaluation results dict.
        model_name: Name of the evaluated model.
        output_path: Optional path to save the report as JSON.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        f"  EVALUATION REPORT: {model_name}",
        "=" * 60,
    ]

    for key, value in results.items():
        if isinstance(value, dict):
            lines.append(f"\n  {key}:")
            for sub_key, sub_val in value.items():
                lines.append(f"    {sub_key}: {sub_val}")
        elif isinstance(value, list) and len(value) > 5:
            lines.append(f"  {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
        else:
            lines.append(f"  {key}: {value}")

    lines.append("=" * 60)
    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Report saved to: {output_path}")

    return report