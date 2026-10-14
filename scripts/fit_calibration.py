"""
BioCellAgents - Fit Confidence Calibration (Temperature Scaling)
=================================================================
Fits the optimal temperature parameter for the organ classifier on the
held-out test set, then saves the calibration parameters.

Usage:
    python scripts/fit_calibration.py
    python scripts/fit_calibration.py --max_samples 5000 --batch_size 64
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PanNukeOrganDataset, ORGAN_TYPES  # noqa: E402
from src.models.organ_classifier import OrganClassifier  # noqa: E402
from src.inference.calibration import TemperatureScaler  # noqa: E402

METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt"
OUTPUT_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "calibration.json"


def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling for organ classifier")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples to use for fitting (0 = full test set)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BioCellAgents - Confidence Calibration Fitter")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Max samples: {args.max_samples or 'full'}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = OrganClassifier(num_classes=len(ORGAN_TYPES))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded ({len(ORGAN_TYPES)} classes)")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset = PanNukeOrganDataset(
        metadata_path=str(METADATA_PATH),
        fold="fold3",
        is_train=False,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    if args.max_samples > 0 and len(dataset) > args.max_samples:
        import random
        indices = random.sample(range(len(dataset)), args.max_samples)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"   Subsampled to {args.max_samples} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Fit temperature
    # ------------------------------------------------------------------
    print(f"\nFitting temperature on {len(dataset)} samples...")
    start = time.time()

    scaler = TemperatureScaler(default_temperature=2.0)
    scaler.fit_from_dataloader(dataloader, model, device, max_batches=200)

    elapsed = time.time() - start
    print(f"Fitting complete in {elapsed:.1f}s")
    print(f"   Temperature (T) = {scaler.temperature:.4f}")
    print(f"   NLL = {scaler.best_nll:.4f}")
    print(f"   Samples used = {scaler.n_samples}")

    # ------------------------------------------------------------------
    # Show before/after comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Before/After Calibration Comparison")
    print("=" * 70)

    # Run a few samples to show the difference
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= 3:
                break
            images = images.to(device)
            logits = model(images)
            raw_probs = torch.softmax(logits, dim=-1)
            calibrated_probs = scaler.calibrate(logits)

            for j in range(min(3, images.shape[0])):
                gt = ORGAN_TYPES[labels[j].item()]
                raw_conf = raw_probs[j].max().item()
                cal_conf = calibrated_probs[j].max().item()
                pred_raw = ORGAN_TYPES[raw_probs[j].argmax().item()]
                pred_cal = ORGAN_TYPES[calibrated_probs[j].argmax().item()]
                print(f"  Sample {j}: GT={gt}, "
                      f"Raw: {pred_raw} ({raw_conf:.4f}), "
                      f"Cal: {pred_cal} ({cal_conf:.4f})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    scaler.save(OUTPUT_PATH)
    print(f"\nCalibration saved to: {OUTPUT_PATH}")
    print(f"   {scaler}")

    print("\n" + "=" * 70)
    print("CALIBRATION FITTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
