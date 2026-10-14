import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data.dataset import (
    PanNukeCellPatchDataset,
    PanNukeSegmentationDataset,
    PanNukeOrganDataset,
    CELL_TYPES,
    ORGAN_TYPES,
)
from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter
from src.models.organ_classifier import OrganClassifier
from src.models.segmentation import build_segmenter


def parse_folds(fold_input):
    if isinstance(fold_input, list):
        return fold_input
    raw_folds = [f.strip() for f in str(fold_input).split(",") if f.strip()]
    return [int(f) if f.isdigit() else f for f in raw_folds]


def load_model_weights(model: nn.Module, model_path: str, device: torch.device):
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint weights from: {model_path}")
    else:
        print(f"No checkpoint found at '{model_path}'. Running evaluation with randomly initialized weights.")
    return model


@torch.no_grad()
def evaluate_classification(model, dataloader, device, target_names):
    model.eval()
    all_preds, all_targets = [], []

    for images, labels in tqdm(dataloader, desc="Evaluating Classification"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)

    return acc, macro_f1, report


@torch.no_grad()
def evaluate_segmentation(model, dataloader, device, num_classes=6):
    model.eval()
    total_dice = 0.0
    total_batches = 0

    for images, masks in tqdm(dataloader, desc="Evaluating Segmentation"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        batch_dice = 0.0
        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (masks == c)
            intersection = (pred_c & target_c).sum().float()
            union = pred_c.sum().float() + target_c.sum().float()

            if union > 0:
                batch_dice += (2.0 * intersection / (union + 1e-6)).item()

        total_dice += (batch_dice / num_classes)
        total_batches += 1

    return total_dice / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BioCellAgents Models")
    parser.add_argument("--task", type=str, required=True, choices=["cell", "organ", "segmentation"])
    parser.add_argument("--metadata_path", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--val_fold", type=str, default="3")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="vits14", choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--arch", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "dinov2"],
                        help="Segmentation architecture (segmentation task only)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_folds = parse_folds(args.val_fold)

    print(f"Evaluation Mode | Task: {args.task.upper()} | Device: {device}")

    if args.task == "cell":
        dataset = PanNukeCellPatchDataset(metadata_path=args.metadata_path, fold=val_folds, is_train=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = DINOv2CellClassifier(model_size=args.model_size, num_classes=len(CELL_TYPES))
        model = load_model_weights(model, args.model_path, device).to(device)

        acc, f1, report = evaluate_classification(model, dataloader, device, CELL_TYPES)
        print(f"\nCell Classification Results:\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f}\n\n{report}")

    elif args.task == "segmentation":
        dataset = PanNukeSegmentationDataset(metadata_path=args.metadata_path, fold=val_folds, is_train=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        num_classes = len(CELL_TYPES) + 1 if "Background" not in CELL_TYPES else len(CELL_TYPES)
        model = build_segmenter(arch=args.arch, num_classes=num_classes, model_size=args.model_size)
        model = load_model_weights(model, args.model_path, device).to(device)

        mean_dice = evaluate_segmentation(model, dataloader, device, num_classes=num_classes)
        print(f"\nSegmentation Results:\nMean Dice Score: {mean_dice:.4f}")

    elif args.task == "organ":
        dataset = PanNukeOrganDataset(metadata_path=args.metadata_path, fold=val_folds, is_train=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = OrganClassifier(num_classes=len(ORGAN_TYPES))
        model = load_model_weights(model, args.model_path, device).to(device)

        acc, f1, report = evaluate_classification(model, dataloader, device, ORGAN_TYPES)
        print(f"\nOrgan Classification Results:\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f}\n\n{report}")


if __name__ == "__main__":
    main()