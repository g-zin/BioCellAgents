import sys
from pathlib import Path

# Resolve project root (BioCellAgents/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PanNukeSegmentationDataset, CELL_TYPES
from src.models.segmentation import build_segmenter
from src.models.losses import CombinedSegmentationLoss
from src.models.base_trainer import BaseTrainer


def parse_fold_arg(fold_arg):
    """
    Parses CLI inputs like '1,2', 1, or ['1', '2'] into a clean list of strings/ints.
    """
    if isinstance(fold_arg, (list, tuple)):
        return list(fold_arg)
    if isinstance(fold_arg, str):
        if "," in fold_arg:
            return [f.strip() for f in fold_arg.split(",") if f.strip()]
        return [fold_arg.strip()]
    return [fold_arg]


def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 Cell Segmentation Model")
    parser.add_argument("--metadata_path", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--train_fold", type=str, default="1,2")
    parser.add_argument("--val_fold", type=str, default="3")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--model_size", type=str, default="vits14", choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--arch", type=str, default="deeplabv3plus",
                        choices=["deeplabv3plus", "dinov2"],
                        help="Segmentation architecture: 'deeplabv3plus' (SOTA dense-prediction, default) "
                             "or 'dinov2' (legacy ViT patch-token segmenter)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_folds = parse_fold_arg(args.train_fold)
    val_folds = parse_fold_arg(args.val_fold)

    train_dataset = PanNukeSegmentationDataset(metadata_path=args.metadata_path, fold=train_folds, is_train=True)
    val_dataset = PanNukeSegmentationDataset(metadata_path=args.metadata_path, fold=val_folds, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=(device.type == "cuda")
    )

    # Background (0) + 5 PanNuke cell categories (1-5)
    num_classes = len(CELL_TYPES) + 1
    model = build_segmenter(
        arch=args.arch,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        model_size=args.model_size,
    ).to(device)
    print(f"Segmentation architecture: {args.arch}")

    criterion = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0)
    
    save_dir = Path(args.output_dir) / f"segmentation_{args.arch}"
    
    # Use BaseTrainer for unified training loop
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        output_dir=str(save_dir),
        task_name="segmentation",
    )
    trainer._num_classes = num_classes

    print(f"\nStarting Segmentation Training ({len(train_dataset)} samples) for {args.epochs} epochs...")
    
    history = trainer.train(epochs=args.epochs, metric_key="dice")
    
    print(f"\nTraining complete! Best Val Mean Dice: {history['best_metric']:.4f}")


if __name__ == "__main__":
    main()