import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from PIL import Image

# Resolve project root (BioCellAgents/) so `src.*` imports work when this file
# is executed directly as a script (e.g. `python src/models/inference_pipeline.py`).
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.calibration import TemperatureScaler
from src.inference.validity_gate import ValidityGate
from src.models.cellpose_segmenter import CellposeSegmenter, SemanticSegmenterAdapter


class TwoStageCellAnalyzer:
    """Combines spatial segmentation and DINOv2 single-cell crop typing.

    Supports two segmenter interfaces:
      - **Torch model** (e.g. ``DINOv2CellSegmenter``, ``DeepLabV3PlusSegmenter``):
        called as ``segmenter(tensor) -> logits``; the analyzer performs
        argmax / thresholding / connected-components internally.
      - **New-style segmenter** (e.g. ``CellposeSegmenter``, ``SemanticSegmenterAdapter``):
        exposes ``.segment(image_pil) -> np.ndarray`` returning a binary mask
        at the original image resolution; the analyzer only does
        connected-components on the returned mask.
    """

    def __init__(self, segmenter_model, classifier_model, device="cuda",
                 calibration_scaler: TemperatureScaler | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Detect segmenter interface: new-style has .segment() and is NOT a torch.nn.Module
        if hasattr(segmenter_model, "segment") and not isinstance(segmenter_model, torch.nn.Module):
            self.segmenter = segmenter_model
            self._segmenter_is_torch = False
        else:
            self.segmenter = segmenter_model.to(self.device).eval()
            self._segmenter_is_torch = True

        self.classifier = classifier_model.to(self.device).eval()
        self.calibration_scaler = calibration_scaler or TemperatureScaler(default_temperature=2.0)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_binary_mask(self, image_pil: Image.Image) -> np.ndarray:
        """Return a binary (H, W) uint8 mask of foreground cells at original resolution."""
        if not self._segmenter_is_torch:
            # New-style segmenter: .segment() returns binary mask at original resolution
            return self.segmenter.segment(image_pil)

        # Torch model path: forward pass -> logits -> argmax -> threshold
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            seg_logits = self.segmenter(input_tensor)  # (1, num_classes, 224, 224)
        pred_class_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        binary_mask_224 = (pred_class_mask > 0).astype(np.uint8)
        # Resize back to original image resolution
        orig_w, orig_h = image_pil.size
        binary_mask = cv2.resize(binary_mask_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return binary_mask

    @torch.no_grad()
    def analyze_patch(self, image_pil: Image.Image, cell_classes: list, padding: int = 10):
        orig_w, orig_h = image_pil.size

        # Step 1: Get binary cell mask (handles both segmenter interfaces)
        binary_mask = self._get_binary_mask(image_pil)

        # Step 2: Extract cell connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

        detected_cells = []

        # Iterate through detected nuclei (index 0 is background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # Suppress tiny noise components
            if area < 10:
                continue

            # Crop individual cell region from original image with padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(orig_w, x + w + padding)
            y2 = min(orig_h, y + h + padding)

            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            # Crop individual cell region from original image
            cell_crop = image_pil.crop((x1, y1, x2, y2))
            cell_tensor = self.transform(cell_crop).unsqueeze(0).to(self.device)

            # Step 3: Classify cell crop
            class_logits = self.classifier(cell_tensor)
            pred_class_idx = torch.argmax(class_logits, dim=1).item()
            
            # Apply temperature scaling calibration
            calibrated_probs = self.calibration_scaler.calibrate(class_logits[0])
            confidence = calibrated_probs[pred_class_idx].item()

            centroid_orig = [float(centroids[i][0]), float(centroids[i][1])]

            detected_cells.append({
                "cell_id": i,
                "bbox": [x1, y1, x2, y2],
                "centroid": centroid_orig,
                "class_name": cell_classes[pred_class_idx] if pred_class_idx < len(cell_classes) else "Unknown",
                "confidence": confidence
            })

        return binary_mask, detected_cells


def _load_weights(model, checkpoint_path, device):
    """Load a checkpoint into a model, tolerating both raw state_dicts and wrapped checkpoints."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at '{checkpoint_path}'. Using randomly initialized weights.")
        return model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    print(f"Loaded weights from: {checkpoint_path}")
    return model


def main():
    """CLI entry point for the two-stage cell analysis inference pipeline (biocell-infer)."""
    import argparse
    import sys
    from pathlib import Path

    # Resolve project root so `src.*` imports work when invoked as a console script.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.dataset import CELL_TYPES
    from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter
    from src.models.segmentation import build_segmenter

    parser = argparse.ArgumentParser(description="Run BioCellAgents two-stage cell analysis inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the input tissue image (PNG/JPG)")
    parser.add_argument("--segmenter_ckpt", type=str,
                        default="checkpoints/segmentation_deeplabv3plus/best_segmentation_model.pt")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="checkpoints/cell_vits14/best_model.pt")
    parser.add_argument("--model_size", type=str, default="vits14",
                        choices=["vits14", "vitb14", "vitl14"])
    parser.add_argument("--arch", type=str, default="deeplabv3plus",
                        choices=["deeplabv3plus", "dinov2", "cellpose"],
                        help="Segmentation architecture used for the spatial pass")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--padding", type=int, default=10, help="Bounding-box padding in pixels")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_cell_classes = len(CELL_TYPES)

    if args.arch == "cellpose":
        # Use pretrained Cellpose-SAM v2 (no checkpoint needed)
        segmenter = CellposeSegmenter(device=str(device))
        print("Using CellposeSegmenter (Cellpose-SAM v2, pretrained)")
    else:
        # Use trained torch model (DeepLabV3+ or DINOv2)
        segmenter = build_segmenter(
            arch=args.arch,
            num_classes=num_cell_classes + 1,
            freeze_backbone=True,
            model_size=args.model_size,
        )
        segmenter = _load_weights(segmenter, args.segmenter_ckpt, device)

    classifier = DINOv2CellClassifier(
        model_size=args.model_size,
        num_classes=num_cell_classes,
        freeze_backbone=True,
    )
    classifier = _load_weights(classifier, args.classifier_ckpt, device)

    analyzer = TwoStageCellAnalyzer(segmenter, classifier, device=args.device)

    image = Image.open(args.image).convert("RGB")
    print(f"Analyzing image: {args.image} ({image.size[0]}x{image.size[1]})")

    mask, detected_cells = analyzer.analyze_patch(image, CELL_TYPES, padding=args.padding)

    print(f"\nDetected {len(detected_cells)} cell(s):")
    for cell in detected_cells:
        print(
            f"  - [{cell['cell_id']}] {cell['class_name']:<12} "
            f"conf={cell['confidence']:.3f} bbox={cell['bbox']} centroid={cell['centroid']}"
        )

    if len(detected_cells) == 0:
        print("  (no cells detected - check the segmentation checkpoint / input image)")

    return detected_cells


if __name__ == "__main__":
    main()