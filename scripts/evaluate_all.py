"""
BioCellAgents - Independent Evaluation Harness
================================================
Evaluates all 3 trained models on the held-out fold3 test set and produces a
complete, reproducible evaluation report.

This is the "Phase 1" deliverable: it loads the existing checkpoints, runs
inference on the held-out test fold, and reports the *actual* metrics (not the
claimed ones) with per-class breakdowns, confusion matrices, and segmentation
visualizations.

Outputs (written to results/):
  - eval_report.json          machine-readable metrics for all tasks
  - eval_report.md            human-readable report
  - confusion_cell.png        cell classification confusion matrix
  - confusion_organ.png       organ classification confusion matrix
  - segmentation_samples/     predicted vs GT mask overlays

Usage:
  python scripts/evaluate_all.py
  python scripts/evaluate_all.py --tasks cell organ segmentation
  python scripts/evaluate_all.py --batch_size 64 --max_samples 20000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup so `src.*` imports resolve regardless of the CWD.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    PanNukeCellPatchDataset,
    PanNukeSegmentationDataset,
    PanNukeOrganDataset,
    CELL_TYPES,
    ORGAN_TYPES,
)
from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter  # noqa: E402
from src.models.organ_classifier import OrganClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS = {
    "cell": PROJECT_ROOT / "checkpoints" / "cell_vits14" / "best_model.pt",
    "organ": PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt",
    "segmentation": PROJECT_ROOT / "checkpoints" / "segmentation_vits14" / "best_segmentation_model.pt",
}

# Number of output classes per task (must match the trained checkpoints).
NUM_CLASSES = {
    "cell": len(CELL_TYPES),              # 5
    "organ": len(ORGAN_TYPES),            # 19 (matches metadata.csv distinct organs)
    "segmentation": len(CELL_TYPES) + 1,  # 6 (background + 5 cell types)
}

CLASS_NAMES = {
    "cell": CELL_TYPES,
    "organ": ORGAN_TYPES,
    "segmentation": ["Background"] + CELL_TYPES,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_state_dict(path: Path, device: torch.device) -> dict:
    """Load a checkpoint and return its model state dict."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return state_dict


def build_model(task: str, device: torch.device) -> nn.Module:
    """Instantiate the correct model architecture for a task and load weights."""
    ckpt_path = CHECKPOINTS[task]
    state_dict = load_state_dict(ckpt_path, device)

    if task == "cell":
        model = DINOv2CellClassifier(model_size="vits14", num_classes=NUM_CLASSES["cell"])
    elif task == "organ":
        model = OrganClassifier(num_classes=NUM_CLASSES["organ"])
    elif task == "segmentation":
        model = DINOv2CellSegmenter(model_size="vits14", num_classes=NUM_CLASSES["segmentation"])
    else:
        raise ValueError(f"Unknown task: {task}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"  Loaded {task} checkpoint: {ckpt_path.name} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def classification_metrics(all_targets, all_preds, class_names):
    """Compute accuracy, F1, per-class P/R/F1, and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    labels = list(range(len(class_names)))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", labels=labels, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=labels)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(all_targets)),
    }


def save_confusion_matrix(cm, class_names, path: Path, title: str) -> None:
    """Render a normalized confusion matrix as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.55), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick = np.arange(len(class_names))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {path.name}")


# ---------------------------------------------------------------------------
# Task evaluators
# ---------------------------------------------------------------------------
def _make_dataloader(dataset, batch_size, num_workers, max_samples):
    if max_samples and len(dataset) > max_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(max_samples)))
        print(f"  (capped to {max_samples} samples)")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_cell(device, batch_size, num_workers, max_samples):
    print("\n" + "=" * 70)
    print("CELL CLASSIFICATION (DINOv2 ViT-S/14, 5 classes)")
    print("=" * 70)

    dataset = PanNukeCellPatchDataset(metadata_path=str(METADATA_PATH), fold="fold3", is_train=False)
    dataloader = _make_dataloader(dataset, batch_size, num_workers, max_samples)
    model = build_model("cell", device)

    all_preds, all_targets = [], []
    t0 = time.time()
    for images, labels in tqdm(dataloader, desc="  cell", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    elapsed = time.time() - t0

    metrics = classification_metrics(all_targets, all_preds, CLASS_NAMES["cell"])
    metrics["inference_seconds"] = round(elapsed, 1)
    metrics["samples_per_second"] = round(len(all_targets) / max(elapsed, 1e-6), 1)

    save_confusion_matrix(
        np.array(metrics["confusion_matrix"]), CLASS_NAMES["cell"],
        RESULTS_DIR / "confusion_cell.png", "Cell Type Classification (fold3)"
    )
    print(f"  Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} "
          f"({metrics['num_samples']} samples, {elapsed:.1f}s)")
    return metrics


@torch.no_grad()
def evaluate_organ(device, batch_size, num_workers, max_samples):
    print("\n" + "=" * 70)
    print("ORGAN CLASSIFICATION (EfficientNet-B0, 19 classes)")
    print("=" * 70)

    dataset = PanNukeOrganDataset(metadata_path=str(METADATA_PATH), fold="fold3", is_train=False)
    dataloader = _make_dataloader(dataset, batch_size, num_workers, max_samples)
    model = build_model("organ", device)

    all_preds, all_targets = [], []
    t0 = time.time()
    for images, labels in tqdm(dataloader, desc="  organ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    elapsed = time.time() - t0

    metrics = classification_metrics(all_targets, all_preds, CLASS_NAMES["organ"])
    metrics["inference_seconds"] = round(elapsed, 1)
    metrics["samples_per_second"] = round(len(all_targets) / max(elapsed, 1e-6), 1)

    absent = [n for n, v in metrics["per_class"].items() if v["support"] == 0]
    if absent:
        metrics["classes_without_test_samples"] = absent
        print(f"   Classes with no fold3 samples: {absent}")

    save_confusion_matrix(
        np.array(metrics["confusion_matrix"]), CLASS_NAMES["organ"],
        RESULTS_DIR / "confusion_organ.png", "Organ Classification (fold3)"
    )
    print(f"  Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} "
          f"({metrics['num_samples']} samples, {elapsed:.1f}s)")
    return metrics


@torch.no_grad()
def evaluate_segmentation(device, batch_size, num_workers, max_samples, num_overlays=6):
    print("\n" + "=" * 70)
    print("CELL SEGMENTATION (DINOv2 ViT-S/14 + decoder, 6 classes)")
    print("=" * 70)

    dataset = PanNukeSegmentationDataset(metadata_path=str(METADATA_PATH), fold="fold3", is_train=False)
    dataloader = _make_dataloader(dataset, batch_size, num_workers, max_samples)
    model = build_model("segmentation", device)
    num_classes = NUM_CLASSES["segmentation"]

    per_class_inter = np.zeros(num_classes)
    per_class_pred = np.zeros(num_classes)
    per_class_gt = np.zeros(num_classes)
    n_samples = 0
    overlay_pool = []  # (image, gt, pred) for a few samples
    total_batch_dice = 0.0
    total_batches = 0

    t0 = time.time()
    for images, masks in tqdm(dataloader, desc="  seg", leave=False):
        images = images.to(device)
        masks = masks.cpu().numpy()  # [B, H, W] class map
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # [B, H, W]

        for c in range(num_classes):
            pred_c = (preds == c)
            gt_c = (masks == c)
            per_class_inter[c] += np.sum(pred_c & gt_c)
            per_class_pred[c] += np.sum(pred_c)
            per_class_gt[c] += np.sum(gt_c)

        # Per-batch mean Dice (canonical metric, matches evaluate.py)
        batch_dice = 0.0
        for c in range(num_classes):
            pred_c = (preds == c)
            gt_c = (masks == c)
            inter = np.sum(pred_c & gt_c)
            union = np.sum(pred_c) + np.sum(gt_c)
            if union > 0:
                batch_dice += (2.0 * inter / (union + 1e-6))
        total_batch_dice += batch_dice / num_classes
        total_batches += 1

        n_samples += len(preds)
        if len(overlay_pool) < num_overlays:
            for i in range(min(len(preds), num_overlays - len(overlay_pool))):
                overlay_pool.append((images[i].cpu(), masks[i], preds[i]))

    elapsed = time.time() - t0

    # Per-class Dice and IoU
    per_class_dice, per_class_iou = [], []
    for c in range(num_classes):
        denom = per_class_pred[c] + per_class_gt[c]
        dice_c = (2.0 * per_class_inter[c] / (denom + 1e-8)) if denom > 0 else 0.0
        union = per_class_pred[c] + per_class_gt[c] - per_class_inter[c]
        iou_c = (per_class_inter[c] / (union + 1e-8)) if union > 0 else 0.0
        per_class_dice.append(round(float(dice_c), 4))
        per_class_iou.append(round(float(iou_c), 4))

    fg = list(range(1, num_classes))  # exclude background
    metrics = {
        "mean_dice": round(total_batch_dice / max(total_batches, 1), 4),
        "foreground_dice": round(float(np.mean(per_class_dice[1:])), 4),
        "mean_iou": round(float(np.mean(per_class_iou)), 4),
        "foreground_iou": round(float(np.mean(per_class_iou[1:])), 4),
        "per_class_dice": {CLASS_NAMES["segmentation"][c]: per_class_dice[c] for c in range(num_classes)},
        "per_class_iou": {CLASS_NAMES["segmentation"][c]: per_class_iou[c] for c in range(num_classes)},
        "num_samples": int(n_samples),
        "inference_seconds": round(elapsed, 1),
    }

    save_segmentation_overlays(overlay_pool, RESULTS_DIR / "segmentation_samples")

    print(f"  Mean Dice: {metrics['mean_dice']:.4f} | Foreground Dice: {metrics['foreground_dice']:.4f} "
          f"({n_samples} samples, {elapsed:.1f}s)")
    return metrics


def save_segmentation_overlays(overlay_pool, out_dir: Path) -> None:
    """Save predicted-vs-GT mask overlays for a few segmentation samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    n = len(overlay_pool)
    for idx, (image, gt, pred) in enumerate(overlay_pool):
        img = image.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        def colorize(mask):
            h, w = mask.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(1, int(mask.max()) + 1):
                rgb[mask == c] = cmap(c % 10)[:3]
            return rgb

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].imshow(colorize(gt))
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(colorize(pred))
        axes[2].set_title("Predicted")
        axes[2].axis("off")
        fig.suptitle(f"Segmentation Sample {idx + 1}/{n}")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx + 1}.png", dpi=120)
        plt.close(fig)
    print(f"   Saved {n} segmentation overlays to {out_dir.name}/")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def build_markdown_report(results: dict) -> str:
    lines = []
    lines.append("# BioCellAgents - Evaluation Report (fold3 held-out test set)\n")
    lines.append(f"_Generated: {results['metadata']['timestamp']} | Device: {results['metadata']['device']}_\n")
    lines.append("## Summary\n")
    lines.append("| Task | Primary Metric | Value | Samples |")
    lines.append("|------|----------------|-------|---------|")
    if "cell" in results["tasks"]:
        c = results["tasks"]["cell"]
        if "accuracy" in c:
            lines.append(f"| Cell Classification | Accuracy | **{c['accuracy']:.4f}** | {c['num_samples']:,} |")
            lines.append(f"| Cell Classification | Macro F1 | {c['macro_f1']:.4f} | |")
    if "organ" in results["tasks"]:
        o = results["tasks"]["organ"]
        if "accuracy" in o:
            lines.append(f"| Organ Classification | Accuracy | **{o['accuracy']:.4f}** | {o['num_samples']:,} |")
            lines.append(f"| Organ Classification | Macro F1 | {o['macro_f1']:.4f} | |")
    if "segmentation" in results["tasks"]:
        s = results["tasks"]["segmentation"]
        if "mean_dice" in s:
            lines.append(f"| Segmentation | Mean Dice | **{s['mean_dice']:.4f}** | {s['num_samples']:,} |")
            lines.append(f"| Segmentation | Foreground Dice | {s['foreground_dice']:.4f} | |")
    lines.append("")

    for task in ("cell", "organ"):
        if task not in results["tasks"] or "per_class" not in results["tasks"][task]:
            continue
        t = results["tasks"][task]
        lines.append(f"## {task.capitalize()} Classification - Per-Class\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|----|---------|")
        for name, v in t["per_class"].items():
            lines.append(f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']:,} |")
        lines.append("")

    if "segmentation" in results["tasks"] and "per_class_dice" in results["tasks"]["segmentation"]:
        s = results["tasks"]["segmentation"]
        lines.append("## Segmentation - Per-Class Dice / IoU\n")
        lines.append("| Class | Dice | IoU |")
        lines.append("|-------|------|-----|")
        for name in CLASS_NAMES["segmentation"]:
            lines.append(f"| {name} | {s['per_class_dice'][name]:.4f} | {s['per_class_iou'][name]:.4f} |")
        lines.append("")

    lines.append("## Artifacts\n")
    lines.append("- `confusion_cell.png` - cell classification confusion matrix")
    lines.append("- `confusion_organ.png` - organ classification confusion matrix")
    lines.append("- `segmentation_samples/` - predicted vs GT mask overlays")
    lines.append("- `eval_report.json` - full machine-readable metrics")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BioCellAgents independent evaluation harness")
    parser.add_argument("--tasks", nargs="+", default=["cell", "organ", "segmentation"],
                        choices=["cell", "organ", "segmentation"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional cap on samples per task (0 = full test set)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BioCellAgents - Independent Evaluation Harness")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size} | Max samples: {args.max_samples or 'full'}")

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "metadata_path": str(METADATA_PATH),
            "fold": "fold3",
            "batch_size": args.batch_size,
            "max_samples": args.max_samples or None,
            "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        },
        "tasks": {},
    }

    evaluators = {
        "cell": evaluate_cell,
        "organ": evaluate_organ,
        "segmentation": evaluate_segmentation,
    }

    for task in args.tasks:
        try:
            results["tasks"][task] = evaluators[task](
                device, args.batch_size, args.num_workers, args.max_samples
            )
        except Exception as e:
            print(f"  {task} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results["tasks"][task] = {"error": str(e)}

    json_path = RESULTS_DIR / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON report saved: {json_path}")

    md_path = RESULTS_DIR / "eval_report.md"
    md_path.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"Markdown report saved: {md_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()