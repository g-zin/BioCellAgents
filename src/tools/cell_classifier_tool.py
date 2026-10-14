# src/tools/cell_classifier_tool.py
import os
import numpy as np
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.models.cell_detector import TrainedCellDetector
from src.inference.validity_gate import ValidityGate


class CellClassifierInput(BaseModel):
    fold_dir: str = Field(..., description="Path to fold directory containing images.npy (and optional masks.npy)")
    patch_index: int = Field(0, description="Index of the image patch to analyze")


class CellTypeClassifierTool(BaseTool):
    name: str = "DINOv2 Cell Type Classifier"
    description: str = (
        "Segments cells with a fine-tuned DINOv2 segmentation model and classifies "
        "each detected cell into Neoplastic, Inflammatory, Connective, Dead, or Epithelial."
    )
    args_schema: Type[BaseModel] = CellClassifierInput

    def _run(self, fold_dir: str, patch_index: int = 0) -> Dict[str, Any]:
        images_path = os.path.join(fold_dir, "images.npy")
        if not os.path.exists(images_path):
            return {"error": f"Images file not found at {images_path}"}

        images = np.load(images_path)
        img_patch = images[patch_index]

        detector = TrainedCellDetector()
        detected_cells, _ = detector.detect_and_classify(img_patch)
        breakdown = detector.class_counts(detected_cells)

        # Apply statistical validity gate
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        validity = gate.full_assessment(
            n_cells=len(detected_cells),
            counts_by_class=breakdown,
        )

        return {
            "total_detected_cells": len(detected_cells),
            "class_breakdown": breakdown,
            "detection_source": "Trained DINOv2 segmentation + classification",
            "validity_assessment": validity,
        }