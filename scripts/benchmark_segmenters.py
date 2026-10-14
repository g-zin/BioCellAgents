"""
Benchmark all four cell/nuclei segmenters on the same held-out fold (fold3).

Why this exists
---------------
The repo has three segmenters (DINOv2, DeepLabV3+, StarDist) and none were
ever benchmarked head-to-head on the same data. This script runs all four
(DINOv2, DeepLabV3+, StarDist, Cellpose) on the identical held-out fold and
reports a fair, apples-to-apples comparison so we can pick the single
canonical segmenter before consolidating the codebase.

Models
------
  1. DINOv2CellSegmenter    -- semantic, 6-class, PanNuke-fine-tuned
  2. DeepLabV3PlusSegmenter -- semantic, 6-class, PanNuke-fine-tuned
  3. StarDist2D             -- instance, zero-shot (pretrained 2D_versatile_he)
  4. Cellpose               -- instance, zero-shot (pretrained nuclei)

Metrics
-------
  - Foreground Dice (any-nucleus vs background): comparable across ALL four.
  - Per-class Dice (6-class): only for the two semantic models (StarDist and
    Cellpose do not assign the 5 PanNuke classes -- they only find nuclei).

Outputs
-------
  - results/segmentation_benchmark.json  (machine-readable)
  - reports/model_scorecard.md           (replaces the benchmark section)

Usage
-----
  python scripts/benchmark_segmenters.py
  python scripts/benchmark_segmenters.py --limit 200   # quick smoke test
  python scripts/benchmark_segmenters.py --fold 3 --batch_size 32
  python scripts/benchmark_segmenters.py --skip cellpose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PanNukeSegmentationDataset  # noqa: E402
from src.models.segmentation import build_segmenter  # noqa: E402

# ImageNet normalization -- MUST match src/data/dataset.py
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NUM_CLASSES = 6  # 0=background, 1-5=cell types

MODEL_NOTES = {
    "DINOv2": "semantic, PanNuke-fine-tuned",
    "DeepLabV3+": "semantic, PanNuke-fine-tuned",
    "StarDist": "instance, zero-shot (2D_versatile_he)",
    "Cellpose": "instance, zero-shot (nuclei)",
}
MODEL_TYPES = {
    "DINOv2": "semantic",
    "DeepLabV3+": "semantic",
    "StarDist": "instance",
    "Cellpose": "instance",
}


def dice_from_counts(inter: float, union: float, eps: float = 1e-6) -> float:
    """Dice from pooled intersection/union counts. Empty-vs-empty = 1.0."""
    if union <= 0:
        return 1.0
    return 2.0 * inter / (union + eps)


def denormalize_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization -> HWC uint8 RGB in [0, 255]."""
    img = img_tensor * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    img = (img * 255).to(torch.uint8)
    return img.permute(1, 2, 0).contiguous().numpy()


@torch.no_grad()
def eval_semantic(model, dataloader, device, num_classes: int = NUM_CLASSES) -> dict:
    """Evaluate a 6-class semantic segmenter. Returns foreground + per-class Dice."""
    model.eval()
    fg_inter = fg_union = 0
    class_inter = [0] * num_classes
    class_union = [0] * num_classes

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        pred_fg = preds != 0
        gt_fg = masks != 0
        fg_inter += int((pred_fg & gt_fg).sum())
        fg_union += int(pred_fg.sum()) + int(gt_fg.sum())

        for c in range(num_classes):
            pc = preds == c
            tc = masks == c
            class_inter[c] += int((pc & tc).sum())
            class_union[c] += int(pc.sum()) + int(tc.sum())

    per_class = [dice_from_counts(class_inter[c], class_union[c]) for c in range(num_classes)]
    return {
        "foreground_dice": dice_from_counts(fg_inter, fg_union),
        "per_class_dice": per_class,
        "mean_class_dice": sum(per_class) / num_classes,
    }


def eval_instance(predict_fn, dataloader) -> dict:
    """Evaluate an instance segmenter (StarDist/Cellpose). Foreground Dice only."""
    fg_inter = fg_union = 0
    for images, masks in dataloader:
        masks_np = masks.numpy()  # (B, H, W)
        for i in range(images.shape[0]):
            img = denormalize_to_uint8(images[i])  # (H, W, 3)
            labels = predict_fn(img)  # (H, W) instance map
            pred_fg = labels != 0
            gt_fg = masks_np[i] != 0
            fg_inter += int(np.count_nonzero(pred_fg & gt_fg))
            fg_union += int(np.count_nonzero(pred_fg)) + int(np.count_nonzero(gt_fg))

    return {"foreground_dice": dice_from_counts(fg_inter, fg_union)}


def build_stardist_predictor(device=None):
    """Return predict(rgb_uint8_hwc) -> instance labels, using the repo's StarDist wrapper."""
    from src.models.nuclei_segmenter import NucleiSegmenter

    seg = NucleiSegmenter(use_stardist_fallback=True)
    seg._init_stardist()  # force load now (fail fast)

    def predict(img):
        return seg.segment_raw_image(img)["labels"]

    return predict


def build_cellpose_predictor(device):
    """Return predict(rgb_uint8_hwc) -> instance labels, using Cellpose v4 (Cellpose-SAM)."""
    from cellpose.models import CellposeModel

    gpu = device.type == "cuda"
    net = CellposeModel(gpu=gpu, pretrained_model="cpsam_v2")

    def predict(img):
        masks, _, _ = net.eval(img, channel_axis=2)
        return masks

    return predict


def load_checkpoint(model, path: str, device) -> torch.nn.Module:
    """Load a repo checkpoint (dict with 'model_state_dict' or a raw state_dict)."""
    p = Path(path)
    if p.exists():
        ckpt = torch.load(p, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"  loaded checkpoint: {p}")
    else:
        print(f"  WARNING: no checkpoint at {p}; using random weights")
    return model


def run_semantic(name: str, arch: str, ckpt_path: str, dataloader, device, model_size: str | None = None) -> dict:
    print(f"=== {name} (semantic, 6-class) ===")
    t0 = time.time()
    kwargs = {"arch": arch, "num_classes": NUM_CLASSES}
    if model_size:
        kwargs["model_size"] = model_size
    model = build_segmenter(**kwargs)
    model = load_checkpoint(model, ckpt_path, device).to(device)
    result = eval_semantic(model, dataloader, device)
    result["time_s"] = round(time.time() - t0, 1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_instance(name: str, build_predictor, dataloader, device) -> dict:
    print(f"=== {name} (instance, zero-shot) ===")
    t0 = time.time()
    try:
        predict = build_predictor(device)
        result = eval_instance(predict, dataloader)
        result["time_s"] = round(time.time() - t0, 1)
        return result
    except Exception as e:  # noqa: BLE001 -- report and continue with other models
        print(f"  {name} failed: {e}")
        return {"error": str(e)}


def build_benchmark_section(payload: dict) -> str:
    """Render the markdown benchmark section."""
    lines = [
        "## Segmentation Benchmark (fold3, head-to-head)",
        "",
        f"_Generated {payload['timestamp']} on {payload['device']} with {payload['num_samples']} samples._",
        "",
        "| Model | Type | Foreground Dice | Mean Class Dice | Notes |",
        "|-------|------|-----------------|-----------------|-------|",
    ]
    for name, r in payload["results"].items():
        if "error" in r:
            lines.append(f"| {name} | {MODEL_TYPES.get(name, '?')} | ERROR | - | {r['error'][:40]} |")
            continue
        fg = r.get("foreground_dice", float("nan"))
        mc = r.get("mean_class_dice", float("nan"))
        mc_str = f"{mc:.4f}" if not np.isnan(mc) else "n/a"
        lines.append(f"| {name} | {MODEL_TYPES.get(name, '?')} | {fg:.4f} | {mc_str} | {MODEL_NOTES.get(name, '')} |")
    lines += [
        "",
        "> **Foreground Dice** is the fair cross-model metric (all four can produce it).",
        "> **Mean Class Dice** only applies to the two semantic models (StarDist/Cellpose don't assign the 5 PanNuke classes).",
        "> StarDist and Cellpose are evaluated **zero-shot** (pretrained); DINOv2 and DeepLabV3+ are **fine-tuned on PanNuke**.",
        "",
    ]
    return "\n".join(lines)


def update_scorecard(payload: dict) -> None:
    """Replace (or append) the benchmark section in reports/model_scorecard.md."""
    scorecard = PROJECT_ROOT / "reports" / "model_scorecard.md"
    marker = "## Segmentation Benchmark (fold3, head-to-head)"

    existing = scorecard.read_text(encoding="utf-8") if scorecard.exists() else ""
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"

    section = build_benchmark_section(payload)
    with open(scorecard, "w", encoding="utf-8") as f:
        f.write(existing + "\n" + section + "\n")
    print(f"Updated {scorecard}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all four segmenters on one fold")
    parser.add_argument("--metadata_path", default="data/processed/metadata.csv")
    parser.add_argument("--fold", default="3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--skip", default="", help="Comma-separated models to skip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load the held-out fold once (identical data for all models)
    dataset = PanNukeSegmentationDataset(
        metadata_path=args.metadata_path, fold=args.fold, is_train=False
    )
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Eval samples: {len(dataset)}\n")

    results: dict = {}
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    # --- 1. DINOv2 (semantic) ---
    if "dinov2" not in skip:
        results["DINOv2"] = run_semantic(
            "DINOv2", "dinov2",
            "checkpoints/segmentation_vits14/best_segmentation_model.pt",
            dataloader, device, model_size="vits14",
        )

    # --- 2. DeepLabV3+ (semantic) ---
    if "deeplabv3plus" not in skip:
        results["DeepLabV3+"] = run_semantic(
            "DeepLabV3+", "deeplabv3plus",
            "checkpoints/segmentation_deeplabv3plus/best_segmentation_model.pt",
            dataloader, device,
        )

    # --- 3. StarDist (instance) ---
    if "stardist" not in skip:
        results["StarDist"] = run_instance("StarDist", build_stardist_predictor, dataloader, device)

    # --- 4. Cellpose (instance) ---
    if "cellpose" not in skip:
        results["Cellpose"] = run_instance("Cellpose", build_cellpose_predictor, dataloader, device)

    # --- Summary ---
    print("\n" + "=" * 62)
    print("SEGMENTATION BENCHMARK (fold3)")
    print("=" * 62)
    print(f"{'Model':<16} {'Fg Dice':>10} {'Mean Class Dice':>18}")
    print("-" * 62)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<16} {'ERROR':>10}  {r['error'][:30]}")
            continue
        fg = r.get("foreground_dice", float("nan"))
        mc = r.get("mean_class_dice", float("nan"))
        mc_str = f"{mc:.4f}" if not np.isnan(mc) else "n/a"
        print(f"{name:<16} {fg:>10.4f} {mc_str:>18}")
    print("=" * 62)

    # --- Write JSON ---
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "segmentation_benchmark.json"
    payload = {
        "fold": args.fold,
        "num_samples": len(dataset),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")

    # --- Update scorecard ---
    update_scorecard(payload)


if __name__ == "__main__":
    main()
