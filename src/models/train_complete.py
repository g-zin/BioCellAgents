"""
Comprehensive Training Pipeline for Cell Segmentation, Classification, and Organ Classification

Pipeline:
1. Train Segmentation Model (DINOv2CellSegmenter) on full tissue patches
2. Train Classification Model (DINOv2CellClassifier) on individual cell crops
3. Train Organ Classification Model (OrganClassifier) on organ-level images
"""

import sys
from pathlib import Path
import os

# Disable xFormers for compatibility
os.environ["XFORMERS_DISABLED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import json
from datetime import datetime

from src.data.dataset import (
    PanNukeSegmentationDataset,
    PanNukeCellPatchDataset,
    PanNukeOrganDataset,
    CELL_TYPES,
    ORGAN_TYPES,
)
from src.models.dino_classifier import DINOv2CellClassifier
from src.models.organ_classifier import OrganClassifier
from src.models.cellpose_segmenter import CellposeSegmenter, SemanticSegmenterAdapter
from src.models.losses import CombinedSegmentationLoss
from src.models.base_trainer import BaseTrainer


# ============================================================================
# SEGMENTATION TRAINING (using BaseTrainer)
# ============================================================================


def train_segmentation_model(args):
    """Train the segmentation model."""
    print("\n" + "="*70)
    print("STEP 1: TRAINING SEGMENTATION MODEL")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PanNukeSegmentationDataset(
        metadata_path=args.metadata_path,
        fold=args.train_fold,
        is_train=True
    )
    val_dataset = PanNukeSegmentationDataset(
        metadata_path=args.metadata_path,
        fold=args.val_fold,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.seg_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.seg_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Load model
    print("\nLoading segmentation model...")
    try:
        from src.models.dino_classifier import DINOv2CellSegmenter
        num_classes = len(CELL_TYPES) + 1
        model = DINOv2CellSegmenter(
            model_size=args.model_size,
            num_classes=num_classes,
            freeze_backbone=args.freeze_backbone
        ).to(device)
        print(f"DINOv2CellSegmenter ({args.model_size}) loaded")
    except ImportError:
        print("DINOv2CellSegmenter not available. Skipping segmentation training.")
        return
    
    # Setup training with BaseTrainer
    criterion = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0)
    
    # Create checkpoint directory
    save_dir = Path(args.output_dir) / f"segmentation_{args.model_size}"
    
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        backbone_lr=args.seg_backbone_lr,
        head_lr=args.seg_head_lr,
        output_dir=str(save_dir),
        task_name="segmentation",
    )
    trainer._num_classes = num_classes

    print(f"\nTraining for {args.seg_epochs} epochs...")
    
    history = trainer.train(
        epochs=args.seg_epochs,
        metric_key="dice",
        patience=args.patience,
        min_delta=args.min_delta,
    )
    
    # Save history
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSegmentation training complete!")
    print(f"   Best Val Dice: {history['best_metric']:.4f}")
    print(f"   Model saved to: {save_dir}")


# ============================================================================
# CLASSIFICATION TRAINING
# ============================================================================

def train_classification_epoch(model, dataloader, criterion, optimizer, device):
    """Train classification model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training Classification", leave=False)
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
def validate_classification(model, dataloader, device):
    """Validate classification model."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    preds_list = []
    labels_list = []
    
    for crops, labels in tqdm(dataloader, desc="Validating Classification", leave=False):
        crops, labels = crops.to(device), labels.to(device)
        logits = model(crops)
        loss = criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * crops.size(0)
        
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = running_loss / total
    
    return avg_loss, accuracy, preds_list, labels_list


def train_classification_model(args):
    """Train the classification model."""
    print("\n" + "="*70)
    print("STEP 2: TRAINING CLASSIFICATION MODEL")
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
        batch_size=args.clf_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.clf_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Train: {len(train_dataset)} cell crops")
    print(f"Val: {len(val_dataset)} cell crops")
    
    # Load model
    print("\nLoading classification model...")
    model = DINOv2CellClassifier(
        model_size=args.model_size,
        num_classes=len(CELL_TYPES),
        freeze_backbone=args.freeze_backbone
    ).to(device)
    print(f"DINOv2CellClassifier ({args.model_size}) loaded")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=args.clf_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.clf_epochs,
        eta_min=1e-7
    )
    
    # Create checkpoint directory
    save_dir = Path(args.output_dir) / f"cell_{args.model_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0
    
    best_val_acc = 0.0
    training_history = defaultdict(list)
    
    print(f"\nTraining for {args.clf_epochs} epochs...")
    
    for epoch in range(1, args.clf_epochs + 1):
        train_loss, train_acc = train_classification_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_classification(model, val_loader, device)
        scheduler.step()
        
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.clf_epochs:02d}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
            }
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ... No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n  [EARLY STOP] triggered after {epoch} epochs "
                      f"(no improvement for {args.patience} epochs)")
                break
    
    # Save history
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nClassification training complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to: {save_dir}")


def train_organ_model(args):
    """Train the organ classification model."""
    print("\n" + "="*70)
    print("STEP 3: TRAINING ORGAN CLASSIFICATION MODEL")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PanNukeOrganDataset(
        metadata_path=args.metadata_path,
        fold=args.train_fold,
        is_train=True
    )
    val_dataset = PanNukeOrganDataset(
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
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Load model
    print("\nLoading organ classification model...")
    model = OrganClassifier(
        num_classes=len(ORGAN_TYPES),
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OrganClassifier")
    print(f"  Total params: {num_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    if args.freeze_backbone:
        # Only train classifier head
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.lr,
            weight_decay=1e-2
        )
        print(f"Training only classifier head (backbone frozen)")
    else:
        # Train entire model with differential LR
        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.01, "weight_decay": 1e-2},
            {"params": model.classifier.parameters(), "lr": args.lr, "weight_decay": 1e-2},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        print(f"Training entire model (backbone LR: {args.lr * 0.01:.2e}, head LR: {args.lr:.2e})")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # Create checkpoint directory
    save_dir = Path(args.output_dir) / f"organ_{args.model_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0
    
    best_val_acc = 0.0
    training_history = defaultdict(list)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"   Learning rate: {args.lr:.2e}")
    print(f"   Batch size: {args.batch_size}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_classification_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_classification(model, val_loader, device)
        scheduler.step()
        
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(float(train_loss))
        training_history["train_acc"].append(float(train_acc))
        training_history["val_loss"].append(float(val_loss))
        training_history["val_acc"].append(float(val_acc))
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": float(val_acc),
                "model_config": {
                    "model_size": args.model_size,
                    "num_classes": len(ORGAN_TYPES),
                    "freeze_backbone": args.freeze_backbone,
                }
            }
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ... No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n  [EARLY STOP] triggered after {epoch} epochs "
                      f"(no improvement for {args.patience} epochs)")
                break
    
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
    parser = argparse.ArgumentParser(description="Train Cell Segmentation, Classification, and Organ Classification Models")
    
    # Common arguments
    parser.add_argument("--metadata_path", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--train_fold", type=str, default="1,2")
    parser.add_argument("--val_fold", type=str, default="3")
    parser.add_argument("--model_size", type=str, default="vits14", choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Segmentation arguments
    parser.add_argument("--seg_epochs", type=int, default=20)
    parser.add_argument("--seg_batch_size", type=int, default=16)
    parser.add_argument("--seg_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--seg_head_lr", type=float, default=3e-4)
    
    # Classification arguments
    parser.add_argument("--clf_epochs", type=int, default=30)
    parser.add_argument("--clf_batch_size", type=int, default=64)
    parser.add_argument("--clf_lr", type=float, default=1e-3)
    
    # Organ classification arguments
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum improvement to reset the early stopping counter")
    
    # Pipeline control
    parser.add_argument("--train_segmentation", action="store_true", default=True)
    parser.add_argument("--train_classification", action="store_true", default=True)
    parser.add_argument("--train_organ", action="store_true", default=True)
    parser.add_argument("--skip_seg", action="store_true", help="Skip segmentation training")
    parser.add_argument("--skip_clf", action="store_true", help="Skip classification training")
    parser.add_argument("--skip_organ", action="store_true", help="Skip organ classification training")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BIOCELL AGENTS: Multi-Agent Training Pipeline")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        if not args.skip_seg and args.train_segmentation:
            train_segmentation_model(args)
        else:
            print("\nSkipping segmentation training")
    
        if not args.skip_clf and args.train_classification:
            train_classification_model(args)
        else:
            print("\nSkipping classification training")
    
        if not args.skip_organ and args.train_organ:
            train_organ_model(args)
        else:
            print("\nSkipping organ classification training")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
