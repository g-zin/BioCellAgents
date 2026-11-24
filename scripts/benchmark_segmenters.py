"""
Benchmark all four cell/nuclei segmenters on the same held-out fold (fold3).

Models:
  1. DINOv2CellSegmenter    -- semantic, 6-class, PanNuke-fine-tuned
  2. DeepLabV3PlusSegmenter -- semantic, 6-class, PanNuke-fine-tuned
  3. StarDist2D             -- instance, zero-shot (pretrained 2D_versatile_he)
  4. Cellpose               -- instance, zero-shot (pretrained Cellpose-SAM v2)

Usage:
  python scripts/benchmark_segmenters.py
  python scripts/benchmark_segmenters.py --limit 200 --skip cellpose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PanNukeSegmentationDataset  # noqa: E402
from src.models.segmentation import build_segmenter  # noqa: E402
from src.utils.device import (  # noqa: E402
    get_device,
    load_checkpoint_safe,
    pin_memory_for_device,
)

# ImageNet normalization -- MUST match src/data/dataset.py
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NUM_CLASSES = 6  # 0=background, 1-5=cell types

MODEL_NOTES = {
    "DINOv2": "semantic, PanNuke-fine-tuned",
    "DeepLabV3+": "semantic, PanNuke-fine-tuned",
    "StarDist": "instance, zero-shot (2D_versatile_he)",
    "Cellpose": "instance, zero-shot (Cellpose-SAM v2)",
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
def eval_semantic(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """Evaluate a 6-class semantic segmenter. Returns foreground + per-class Dice."""
    model.eval()
    fg_inter = fg_union = 0
    class_inter = [0] * num_classes
    class_union = [0] * num_classes
    non_blocking = pin_memory_for_device(device)

    for images, masks in dataloader:
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        pred_fg = preds != 0
        gt_fg = masks != 0
        fg_inter += int((pred_fg & gt_fg).sum().item())
        fg_union += int(pred_fg.sum().item()) + int(gt_fg.sum().item())

        for c in range(num_classes):
            pc = preds == c
            tc = masks == c
            class_inter[c] += int((pc & tc).sum().item())
            class_union[c] += int(pc.sum().item()) + int(tc.sum().item())

    per_class = [dice_from_counts(class_inter[c], class_union[c]) for c in range(num_classes)]
    return {
        "foreground_dice": dice_from_counts(fg_inter, fg_union),
        "per_class_dice": per_class,
        "mean_class_dice": sum(per_class) / num_classes,
    }


def eval_instance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    dataloader: DataLoader,
) -> Dict[str, Any]:
    """Evaluate an instance segmenter (StarDist/Cellpose). Foreground Dice only."""
    fg_inter = fg_union = 0
    for images, masks in dataloader:
        masks_np = masks.numpy()
        for i in range(images.shape[0]):
            img = denormalize_to_uint8(images[i])
            labels = predict_fn(img)
            pred_fg = labels != 0
            gt_fg = masks_np[i] != 0
            fg_inter += int(np.count_nonzero(pred_fg & gt_fg))
            fg_union += int(np.count_nonzero(pred_fg)) + int(np.count_nonzero(gt_fg))

    return {"foreground_dice": dice_from_counts(fg_inter, fg_union)}


def build_stardist_predictor(
    device: Optional[torch.device] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return predict(rgb_uint8_hwc) -> instance labels, using the repo's StarDist wrapper."""
    from src.models.nuclei_segmenter import NucleiSegmenter

    _ = device
    seg = NucleiSegmenter(use_stardist_fallback=True)
    seg._init_stardist()

    def predict(img: np.ndarray) -> np.ndarray:
        return np.asarray(seg.segment_raw_image(img)["labels"])

    return predict


def build_cellpose_predictor(device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Return predict(rgb_uint8_hwc) -> mask, using the unified CellposeSegmenter wrapper."""
    from PIL import Image

    from src.models.cellpose_segmenter import CellposeSegmenter

    segmenter = CellposeSegmenter(device=str(device))

    def predict(img: np.ndarray) -> np.ndarray:
        return segmenter.segment(Image.fromarray(img))

    return predict


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Load a repo checkpoint safely using ``load_checkpoint_safe``."""
    p = Path(path)
    if p.exists():
        state_dict = load_checkpoint_safe(p, map_location=device, extract_state_dict=True)
        model.load_state_dict(state_dict)
        print(f"  loaded checkpoint: {p}")
    else:
        print(f"  WARNING: no checkpoint at {p}; using random weights")
    return model


def run_semantic(
    name: str,
    arch: str,
    ckpt_path: str,
    dataloader: DataLoader,
    device: torch.device,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"=== {name} (semantic, 6-class) ===")
    t0 = time.time()
    kwargs: Dict[str, Any] = {"arch": arch, "num_classes": NUM_CLASSES}
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


def run_instance(
    name: str,
    build_predictor: Callable[[torch.device], Callable[[np.ndarray], np.ndarray]],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"=== {name} (instance, zero-shot) ===")
    t0 = time.time()
    try:
        predict = build_predictor(device)
        result = eval_instance(predict, dataloader)
        result["time_s"] = round(time.time() - t0, 1)
        return result
    except Exception as exc:
        print(f"  {name} failed: {exc}")
        return {"error": str(exc)}


def build_benchmark_section(payload: Dict[str, Any]) -> str:
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
            lines.append(
                f"| {name} | {MODEL_TYPES.get(name, '?')} | ERROR | - | {r['error'][:40]} |"
            )
            continue
        fg = float(r.get("foreground_dice", float("nan")))
        mc = float(r.get("mean_class_dice", float("nan")))
        mc_str = f"{mc:.4f}" if not np.isnan(mc) else "n/a"
        lines.append(
            f"| {name} | {MODEL_TYPES.get(name, '?')} | {fg:.4f} | {mc_str} | "
            f"{MODEL_NOTES.get(name, '')} |"
        )
    lines += [
        "",
        "> **Foreground Dice** is the fair cross-model metric (all four can produce it).",
        "> **Mean Class Dice** only applies to the two semantic models "
        "(StarDist/Cellpose don't assign the 5 PanNuke classes).",
        "> StarDist and Cellpose are evaluated **zero-shot** (pretrained); "
        "DINOv2 and DeepLabV3+ are **fine-tuned on PanNuke**.",
        "",
    ]
    return "\n".join(lines)


def update_scorecard(payload: Dict[str, Any]) -> None:
    """Replace (or append) the benchmark section in reports/model_scorecard.md."""
    scorecard = PROJECT_ROOT / "reports" / "model_scorecard.md"
    scorecard.parent.mkdir(parents=True, exist_ok=True)
    marker = "## Segmentation Benchmark (fold3, head-to-head)"

    existing = scorecard.read_text(encoding="utf-8") if scorecard.exists() else ""
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"

    section = build_benchmark_section(payload)
    scorecard.write_text(existing + "\n" + section + "\n", encoding="utf-8")
    print(f"Updated {scorecard}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark all four segmenters on one fold")
    parser.add_argument("--metadata_path", default="data/processed/metadata.csv")
    parser.add_argument("--fold", default="fold3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--skip", default="", help="Comma-separated models to skip")
    parser.add_argument("--device", default=None, help="Preferred torch device")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    print(f"Device: {device}\n")

    dataset: Dataset[Any] = PanNukeSegmentationDataset(
        metadata_path=args.metadata_path, fold=args.fold, is_train=False
    )
    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))  # type: ignore[arg-type]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory_for_device(device),
    )
    num_samples = len(dataset)  # type: ignore[arg-type]
    print(f"Eval samples: {num_samples}\n")

    results: Dict[str, Any] = {}
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    if "dinov2" not in skip:
        results["DINOv2"] = run_semantic(
            "DINOv2",
            "dinov2",
            "checkpoints/segmentation_vits14/best_segmentation_model.pt",
            dataloader,
            device,
            model_size="vits14",
        )

    if "deeplabv3plus" not in skip:
        results["DeepLabV3+"] = run_semantic(
            "DeepLabV3+",
            "deeplabv3plus",
            "checkpoints/segmentation_deeplabv3plus/best_segmentation_model.pt",
            dataloader,
            device,
        )

    if "stardist" not in skip:
        results["StarDist"] = run_instance("StarDist", build_stardist_predictor, dataloader, device)

    if "cellpose" not in skip:
        results["Cellpose"] = run_instance("Cellpose", build_cellpose_predictor, dataloader, device)

    print("\n" + "=" * 62)
    print(f"SEGMENTATION BENCHMARK ({args.fold})")
    print("=" * 62)
    print(f"{'Model':<16} {'Fg Dice':>10} {'Mean Class Dice':>18}")
    print("-" * 62)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<16} {'ERROR':>10}  {r['error'][:30]}")
            continue
        fg = float(r.get("foreground_dice", float("nan")))
        mc = float(r.get("mean_class_dice", float("nan")))
        mc_str = f"{mc:.4f}" if not np.isnan(mc) else "n/a"
        print(f"{name:<16} {fg:>10.4f} {mc_str:>18}")
    print("=" * 62)

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segmentation_benchmark.json"
    payload = {
        "fold": args.fold,
        "num_samples": num_samples,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    update_scorecard(payload)


if __name__ == "__main__":
    main()
