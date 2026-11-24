"""Diagnostic: compare global vs per-batch Dice metrics on a segmentation model.

- Global Dice: pools intersections/unions across the whole validation set,
  computes per-class Dice from global sums, then macro-averages over present classes.
- Per-batch mean Dice: computes per-batch Dice (summing present classes, divided
  by ``num_classes``), then averages across batches (matches ``evaluate.py``).

Usage:
    python scripts/diag_dice.py
    python scripts/diag_dice.py --arch deeplabv3plus --fold fold3 --batch_size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.dataset import CELL_TYPES, PanNukeSegmentationDataset  # noqa: E402
from src.models.segmentation import build_segmenter  # noqa: E402
from src.utils.device import (  # noqa: E402
    get_device,
    load_checkpoint_safe,
    pin_memory_for_device,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare global vs per-batch Dice metrics on a segmentation fold"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/segmentation_deeplabv3plus/best_segmentation_model.pt",
        help="Path to segmentation checkpoint",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="deeplabv3plus",
        choices=["deeplabv3plus", "dinov2"],
        help="Segmentation model architecture",
    )
    parser.add_argument("--model_size", type=str, default="vits14")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/processed/metadata.csv",
        help="Path to PanNuke metadata CSV",
    )
    parser.add_argument("--fold", type=str, default="fold3", help="Evaluation fold")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    num_classes = len(CELL_TYPES) + 1
    class_names = ["Background"] + CELL_TYPES

    dataset = PanNukeSegmentationDataset(
        metadata_path=args.metadata_path, fold=args.fold, is_train=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory_for_device(device),
    )

    model = build_segmenter(arch=args.arch, num_classes=num_classes, model_size=args.model_size)
    ckpt_path = Path(args.model_path)
    if ckpt_path.exists():
        state_dict = load_checkpoint_safe(ckpt_path, map_location=device, extract_state_dict=True)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}; evaluating random weights.")
    model.to(device).eval()
    print(
        f"Loaded {args.arch} from {args.model_path} on {device} | "
        f"{len(dataset)} val samples | {num_classes} classes"
    )

    g_inter = torch.zeros(num_classes, device=device)
    g_union = torch.zeros(num_classes, device=device)
    total_batch_dice = 0.0
    total_batches = 0
    pc_sum = [0.0] * num_classes
    pc_cnt = [0] * num_classes

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Diag"):
            images = images.to(device, non_blocking=pin_memory_for_device(device))
            masks = masks.to(device, non_blocking=pin_memory_for_device(device))
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            batch_dice = 0.0
            for c in range(num_classes):
                pred_c = preds == c
                target_c = masks == c
                inter = (pred_c & target_c).sum().float()
                union = pred_c.sum().float() + target_c.sum().float()
                g_inter[c] += inter
                g_union[c] += union
                if union > 0:
                    d = float((2.0 * inter / (union + 1e-6)).item())
                    batch_dice += d
                    pc_sum[c] += d
                    pc_cnt[c] += 1
            total_batch_dice += batch_dice / num_classes
            total_batches += 1

    g_dice = (2.0 * g_inter) / (g_union + 1e-6)
    valid = g_union > 0
    global_macro = float(g_dice[valid].mean().item()) if bool(valid.any()) else 0.0
    perbatch_macro = total_batch_dice / max(total_batches, 1)
    pc_mean = [pc_sum[c] / pc_cnt[c] if pc_cnt[c] > 0 else float("nan") for c in range(num_classes)]

    print("\n" + "=" * 62)
    print(f"{'Class':<15} {'Global Dice':>13} {'PerBatch Dice':>14} {'#batches':>10}")
    print("-" * 62)
    for c in range(num_classes):
        print(f"{class_names[c]:<15} {g_dice[c].item():>13.4f} {pc_mean[c]:>14.4f} {pc_cnt[c]:>10}")
    print("-" * 62)
    print(f"{'MACRO avg':<15} {global_macro:>13.4f} {perbatch_macro:>14.4f}")
    print("=" * 62)
    print(f"\nGlobal Dice (macro over present classes): {global_macro:.4f}")
    print(f"Per-batch mean Dice (evaluate.py style):  {perbatch_macro:.4f}")


if __name__ == "__main__":
    main()
