"""Diagnostic: compare the two Dice metrics used in the codebase.

- Trainer-style GLOBAL Dice (base_trainer.validate_epoch): pools intersections/unions
  across the whole validation set, computes per-class Dice from the global sums, then
  macro-averages over classes that are present.
- evaluate.py PER-BATCH mean Dice: computes per-batch Dice (summing only present
  classes but dividing by num_classes), then averages over batches.

Both are run on the SAME model + SAME validation fold so the only difference is the
metric definition.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PanNukeSegmentationDataset, CELL_TYPES
from src.models.segmentation import build_segmenter

MODEL_PATH = "checkpoints/segmentation_deeplabv3plus/best_segmentation_model.pt"
ARCH = "deeplabv3plus"
FOLD = ["fold3"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CELL_TYPES) + 1  # 6 (Background + 5 cell types)
class_names = ["Background"] + CELL_TYPES

dataset = PanNukeSegmentationDataset(metadata_path="data/processed/metadata.csv", fold=FOLD, is_train=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

model = build_segmenter(arch=ARCH, num_classes=num_classes, model_size="vits14")
ckpt = torch.load(MODEL_PATH, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.to(device).eval()
print(f"Loaded {ARCH} from {MODEL_PATH} | {len(dataset)} val samples | {num_classes} classes")

# --- Accumulators ---
g_inter = torch.zeros(num_classes, device=device)   # global intersection per class
g_union = torch.zeros(num_classes, device=device)   # global (pred+target) per class
total_batch_dice = 0.0                              # evaluate.py style
total_batches = 0
pc_sum = [0.0] * num_classes                        # per-class per-batch dice sum
pc_cnt = [0] * num_classes                          # per-class per-batch dice count

with torch.no_grad():
    for images, masks in tqdm(dataloader, desc="Diag"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        batch_dice = 0.0
        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (masks == c)
            inter = (pred_c & target_c).sum().float()
            union = pred_c.sum().float() + target_c.sum().float()
            g_inter[c] += inter
            g_union[c] += union
            if union > 0:
                d = (2.0 * inter / (union + 1e-6)).item()
                batch_dice += d
                pc_sum[c] += d
                pc_cnt[c] += 1
        total_batch_dice += batch_dice / num_classes  # NOTE: divides by num_classes (6)
        total_batches += 1

g_dice = (2.0 * g_inter) / (g_union + 1e-6)
valid = g_union > 0
global_macro = g_dice[valid].mean().item()
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
print(f"\nTrainer-style GLOBAL Dice (macro over present classes): {global_macro:.4f}")
print(f"evaluate.py PER-BATCH mean Dice (sum present / num_classes): {perbatch_macro:.4f}")
print(f"\n=> The training log's 'Val DICE' uses the GLOBAL metric; evaluate.py uses PER-BATCH.")
