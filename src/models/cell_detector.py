"""Shared model-driven cell detection + classification.

Wraps the trained DINOv2 segmentation + classification checkpoints so the CrewAI
tools (spatial graph analytics, cell type classifier) can detect and type cells
from the raw H&E patch instead of falling back to ground-truth masks.

The two-stage pipeline:
  1. DINOv2CellSegmenter  -> per-pixel class map -> connected components -> centroids
  2. DINOv2CellClassifier -> per-cell type (Neoplastic / Inflammatory / ...)
"""
import os

os.environ["XFORMERS_DISABLED"] = "1"

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.data.dataset import CELL_TYPES
from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter
from src.models.inference_pipeline import TwoStageCellAnalyzer
from src.inference.calibration import TemperatureScaler


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image patch to a uint8 RGB array (H x W x 3)."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


class TrainedCellDetector:
    """Detects and classifies cells using the trained DINOv2 checkpoints.

    Models are loaded lazily on first use and cached on the instance.
    """

    def __init__(
        self,
        model_size: str = "vits14",
        segmenter_ckpt: str = "checkpoints/segmentation_vits14/best_segmentation_model.pt",
        classifier_ckpt: str = "checkpoints/cell_vits14/best_model.pt",
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.segmenter_ckpt = segmenter_ckpt
        self.classifier_ckpt = classifier_ckpt
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._analyzer: Optional[TwoStageCellAnalyzer] = None

    def _load_weights(self, model, checkpoint_path: str):
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        return model

    def _ensure_loaded(self) -> TwoStageCellAnalyzer:
        if self._analyzer is not None:
            return self._analyzer

        num_cell_classes = len(CELL_TYPES)
        segmenter = DINOv2CellSegmenter(
            model_size=self.model_size,
            num_classes=num_cell_classes + 1,
            freeze_backbone=True,
        )
        segmenter = self._load_weights(segmenter, self.segmenter_ckpt)

        classifier = DINOv2CellClassifier(
            model_size=self.model_size,
            num_classes=num_cell_classes,
            freeze_backbone=True,
        )
        classifier = self._load_weights(classifier, self.classifier_ckpt)

        # Load calibration parameters if available
        cal_path = Path(self.classifier_ckpt).parent / "calibration.json"
        calibration_scaler = TemperatureScaler.load(cal_path)

        self._analyzer = TwoStageCellAnalyzer(
            segmenter, classifier, device=str(self.device),
            calibration_scaler=calibration_scaler,
        )
        return self._analyzer

    def detect_and_classify(self, image: np.ndarray, padding: int = 10) -> tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect + classify cells in an image patch.

        Args:
            image: H x W x 3 (or H x W) image patch, any dtype.
            padding: bounding-box padding in 224-space pixels.

        Returns:
            Tuple of (detected_cells, mask):
              - detected_cells: List of dicts {cell_id, bbox, centroid: [x, y], class_name, confidence}
              - mask: Binary (H, W) uint8 segmentation mask at original image resolution
        """
        analyzer = self._ensure_loaded()
        image_pil = Image.fromarray(to_uint8_rgb(image)).convert("RGB")
        mask, detected_cells = analyzer.analyze_patch(image_pil, CELL_TYPES, padding=padding)
        return detected_cells, mask

    def to_centroids(self, detected_cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert detected cells to the centroid dict format used by the spatial graph."""
        centroids = []
        for i, cell in enumerate(detected_cells, 1):
            cx, cy = cell["centroid"]
            centroids.append({
                "id": cell.get("cell_id", i),
                "y": float(cy),
                "x": float(cx),
                "class": cell.get("class_name", "Unknown"),
                "area_px": -1,
            })
        return centroids

    def class_counts(self, detected_cells: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate detected cells into a per-class count dict (all classes present)."""
        counts = {name: 0 for name in CELL_TYPES}
        for cell in detected_cells:
            name = cell.get("class_name", "Unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts