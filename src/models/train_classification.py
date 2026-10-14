"""
Training Pipeline for Cell Classification Model

This script trains DINOv2CellClassifier on individual cell crops.

The segmentation model is not trained here - instead we use:
1. Ground-truth PanNuke masks, or
2. StarDist2D for automatic segmentation

Pipeline:
1. Extract cell crops from images using CellposeSegmenter (with ground-truth masks)
2. Train DINOv2CellClassifier to classify individual cells
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json
from datetime import datetime

from src.data.dataset import PanNukeCellPatchDataset, CELL_TYPES
from src.models.dino_classifier import DINOv2CellClassifier


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for crops, labels in pbar:
        crops, labels = crops.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(crops)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        running_loss += loss.item() * crops.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for crops, labels in tqdm(dataloader, desc="Validating", leave=False):
        crops, labels = crops.to(device), labels.to(device)
        logits = model(crops)
        loss = criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * crops.size(0)
    
    accuracy = correct / total
    avg_loss = running_loss / total
    return avg_loss, accuracy


def train_classification(args):
    """Train the classification model."""
    print("\n" + "="*70)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PanNukeCellPatchDataset(
        metadata_path=args.metadata_path,
        fold=args.train_fold,
        is_train=True
    )
    val_dataset = PanNukeCellPatchDataset(
        metadata_path=args.metadata_path,
        fold=args.val_fold,
        is_train=False
    )
    
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
    
    print(f"Train: {len(train_dataset)} cell crops")
    print(f"Val: {len(val_dataset)} cell crops")
    print(f"Classes: {len(CELL_TYPES)} ({', '.join(CELL_TYPES)})")
    
    # Load model
    print("\nLoading classification model...")
    model = DINOv2CellClassifier(
        model_size=args.model_size,
        num_classes=len(CELL_TYPES),
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DINOv2CellClassifier ({args.model_size})")
    print(f"  Total params: {num_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    if args.freeze_backbone:
        # Only train classifier head
        optimizer = torch.optim.AdamW(
            model.classifier_head.parameters(),
            lr=args.lr,
            weight_decay=1e-2
        )
        print(f"Training only classifier head (backbone frozen)")
    else:
        # Train entire model with differential LR
        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.01, "weight_decay": 1e-2},
            {"params": model.classifier_head.parameters(), "lr": args.lr, "weight_decay": 1e-2},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        print(f"Training entire model (backbone LR: {args.lr * 0.01:.2e}, head LR: {args.lr:.2e})")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # Create checkpoint directory
    save_dir = Path(args.output_dir) / f"cell_{args.model_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    training_history = defaultdict(list)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"   Learning rate: {args.lr:.2e}")
    print(f"   Batch size: {args.batch_size}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(float(train_loss))
        training_history["train_acc"].append(float(train_acc))
        training_history["val_loss"].append(float(val_loss))
        training_history["val_acc"].append(float(val_acc))
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": float(val_acc),
                "model_config": {
                    "model_size": args.model_size,
                    "num_classes": len(CELL_TYPES),
                    "freeze_backbone": args.freeze_backbone,
                }
            }
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
    
    # Save history
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    
    print(f"\nTraining complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")
    print(f"   - best_model.pt (epoch {training_history['epoch'][training_history['val_acc'].index(best_val_acc)]})")
    print(f"   - final_model.pt (epoch {args.epochs})")
    print(f"   - training_history.json")


def main():
    parser = argparse.ArgumentParser(
        description="Train Cell Classification Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with defaults
  python src/models/train_complete.py
  
  # Custom configuration
  python src/models/train_complete.py --epochs 50 --batch_size 128 --model_size vitb14
  
  # Train head only (faster)
  python src/models/train_complete.py --freeze_backbone
  
  # Custom folds
  python src/models/train_complete.py --train_fold "1" --val_fold "2"
        """
    )
    
    # Dataset arguments
    parser.add_argument("--metadata_path", type=str, default="data/processed/metadata.csv",
                        help="Path to metadata CSV file")
    parser.add_argument("--train_fold", type=str, default="1,2",
                        help="Training folds (comma-separated or single number)")
    parser.add_argument("--val_fold", type=str, default="3",
                        help="Validation fold")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="vits14", 
                        choices=["vits14", "vitb14", "vitl14"],
                        help="DINOv2 model size")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze backbone and only train classifier head")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BIOCELL AGENTS: Classification Training")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        train_classification(args)
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
