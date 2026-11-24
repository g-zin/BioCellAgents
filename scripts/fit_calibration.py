"""
BioCellAgents - Fit Confidence Calibration (Temperature Scaling)
=================================================================
Fits the optimal temperature parameter for the organ classifier on a
validation fold (default ``fold2`` to avoid test-set leakage on ``fold3``),
reports raw vs. calibrated Expected Calibration Error (ECE), and saves
calibration parameters.

Usage:
    python scripts/fit_calibration.py
    python scripts/fit_calibration.py --fold fold2 --max_samples 5000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ORGAN_TYPES, PanNukeOrganDataset  # noqa: E402
from src.inference.calibration import TemperatureScaler, compute_ece  # noqa: E402
from src.models.organ_classifier import OrganClassifier  # noqa: E402
from src.utils.device import (  # noqa: E402
    get_device,
    load_checkpoint_safe,
    pin_memory_for_device,
)

METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt"
OUTPUT_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "calibration.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for organ classifier")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=str(METADATA_PATH),
        help="Path to PanNuke metadata CSV",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to organ classifier checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(OUTPUT_PATH),
        help="Destination path for calibration.json",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold2",
        help="Validation fold to fit temperature on (prefer fold2 over held-out test fold3)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Max samples to use for fitting (0 = full fold)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint_path)
    out_path = Path(args.output_path)

    print("=" * 70)
    print("BioCellAgents - Confidence Calibration Fitter")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Fold:        {args.fold}")
    print(f"Max samples: {args.max_samples or 'full'} (seed={args.seed})")

    if str(args.fold).lower() in {"3", "fold3"}:
        print(
            "WARNING: Fitting temperature scaling on held-out test fold3 risks calibration leakage."
        )

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    state_dict = load_checkpoint_safe(ckpt_path, map_location=device, extract_state_dict=True)
    model = OrganClassifier(num_classes=len(ORGAN_TYPES), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Model loaded ({len(ORGAN_TYPES)} classes)")

    dataset: Dataset[Any] = PanNukeOrganDataset(
        metadata_path=args.metadata_path,
        fold=args.fold,
        is_train=False,
    )
    total_available = len(dataset)  # type: ignore[arg-type]
    print(f"Dataset loaded: {total_available} samples")

    if args.max_samples > 0 and total_available > args.max_samples:
        rng = random.Random(args.seed)
        indices = rng.sample(range(total_available), args.max_samples)
        dataset = Subset(dataset, indices)
        print(f"Subsampled deterministically to {args.max_samples} samples (seed={args.seed})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory_for_device(device),
    )

    # Collect logits once so we can fit and measure exact before/after ECE
    all_logits = []
    all_labels = []
    non_blocking = pin_memory_for_device(device)
    start = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=non_blocking)
            all_logits.append(model(images).cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    raw_probs = torch.softmax(logits_tensor, dim=-1)
    ece_before = compute_ece(raw_probs, labels_tensor)

    scaler = TemperatureScaler(default_temperature=2.0)
    scaler.fit(logits_tensor, labels_tensor)

    cal_probs = scaler.calibrate(logits_tensor)
    ece_after = compute_ece(cal_probs, labels_tensor)
    elapsed = time.time() - start

    print(f"\nFitting complete in {elapsed:.1f}s")
    print(f"   Temperature (T) = {scaler.temperature:.4f}")
    print(f"   NLL             = {scaler.best_nll:.4f}")
    print(f"   ECE (before)    = {ece_before:.4f}")
    print(f"   ECE (after)     = {ece_after:.4f}")
    print(f"   Samples used    = {scaler.n_samples}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(out_path)
    print(f"\nCalibration saved to: {out_path}")
    print("CALIBRATION FITTING COMPLETE")


if __name__ == "__main__":
    main()
