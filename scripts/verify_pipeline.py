"""BioCellAgents — End-to-End Pipeline & Hardware Verification Script.

Usage:
    python scripts/verify_pipeline.py             # GPU/CPU pretrained backbone check
    python scripts/verify_pipeline.py --offline   # Zero-network CI/offline verification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.evals.report_fidelity_eval import evaluate_report_fidelity  # noqa: E402
from src.inference.calibration import TemperatureScaler  # noqa: E402
from src.inference.validity_gate import ValidityGate  # noqa: E402
from src.models.dino_classifier import (  # noqa: E402
    DINOv2CellClassifier,
    DINOv2CellSegmenter,
)
from src.models.organ_classifier import OrganClassifier  # noqa: E402
from src.reporting import render_deterministic_report  # noqa: E402
from src.tools.graph_analytics import SpatialGraphAnalytics  # noqa: E402
from src.utils.device import get_device, get_hardware_info  # noqa: E402


class _OfflineDINOBackbone(nn.Module):
    """Lightweight deterministic DINOv2-compatible backbone for --offline verification."""

    def __init__(self, embed_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        return feat.mean(dim=(2, 3))

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.proj(x)
        tokens = feat.flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify BioCellAgents models, spatial, and reporting"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with offline lightweight backbones (zero internet/hub downloads)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Preferred torch device (auto-detects cuda -> mps -> cpu if omitted)",
    )
    return parser


def test_pipeline(offline: bool = False, preferred_device: str | None = None) -> bool:
    device = get_device(preferred_device)
    hw = get_hardware_info()
    print(f"BioCellAgents Verification | Device: {device} | Hardware: {hw['device_name']}")
    print(f"PyTorch: {hw['pytorch_version']} | CUDA available: {hw['cuda_available']}\n")

    # 1. Single-Cell Classifier
    print("1. Testing DINOv2CellClassifier...")
    cls_backbone = _OfflineDINOBackbone(384) if offline else None
    classifier = DINOv2CellClassifier(model_size="vits14", num_classes=5, backbone=cls_backbone).to(
        device
    )
    classifier.eval()
    dummy_cls = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        cls_out = classifier(dummy_cls)
    assert cls_out.shape == (2, 5), f"Unexpected classifier shape: {cls_out.shape}"
    # Verify CPU crop list -> device auto-placement works without device mismatch
    crop_preds = classifier.classify_crops([torch.randn(3, 64, 64), torch.randn(3, 64, 64)])
    assert len(crop_preds) == 2
    print(f"   PASS: logits={tuple(cls_out.shape)}, classify_crops={len(crop_preds)} crops")

    # 2. Spatial Segmenter
    print("2. Testing DINOv2CellSegmenter...")
    seg_backbone = _OfflineDINOBackbone(384) if offline else None
    segmenter = DINOv2CellSegmenter(model_size="vits14", num_classes=6, backbone=seg_backbone).to(
        device
    )
    segmenter.eval()
    dummy_seg = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        seg_out = segmenter(dummy_seg)
    assert seg_out.shape == (2, 6, 224, 224), f"Unexpected segmenter shape: {seg_out.shape}"
    print(f"   PASS: logits={tuple(seg_out.shape)}")

    # 3. Organ Classifier + Calibration
    print("3. Testing OrganClassifier + TemperatureScaler...")
    organ_model = OrganClassifier(num_classes=19, pretrained=not offline).to(device)
    organ_model.eval()
    with torch.no_grad():
        organ_logits = organ_model(dummy_cls)
    assert organ_logits.shape == (2, 19), f"Unexpected organ shape: {organ_logits.shape}"
    scaler = TemperatureScaler(default_temperature=2.0)
    cal_probs = scaler.calibrate(organ_logits[0])
    assert abs(float(cal_probs.sum().item()) - 1.0) < 1e-4
    print(
        f"   PASS: logits={tuple(organ_logits.shape)}, calibrated_sum={cal_probs.sum().item():.4f}"
    )

    # 4. Spatial Graph + Validity Gate + Deterministic Reporting + Fidelity Audit
    print("4. Testing SpatialGraphAnalytics + ValidityGate + Reporting + Fidelity Grader...")
    centroids = [
        {"id": i, "x": float(20 + (i % 6) * 30), "y": float(20 + (i // 6) * 30), "class": cls}
        for i, cls in enumerate(
            (["Neoplastic"] * 14)
            + (["Inflammatory"] * 8)
            + (["Connective"] * 6)
            + (["Epithelial"] * 5)
            + (["Dead"] * 3)
        )
    ]
    analyzer = SpatialGraphAnalytics(max_distance_px=50.0)
    graph_metrics = analyzer.build_graph_and_compute_metrics(centroids, patch_index=999)
    counts = {"Neoplastic": 14, "Inflammatory": 8, "Connective": 6, "Epithelial": 5, "Dead": 3}
    gate = ValidityGate(min_total_cells=30, min_per_class=5)
    validity = gate.full_assessment(n_cells=len(centroids), counts_by_class=counts)
    payload = {
        "schema_version": "2.0",
        "patch_index": 999,
        "fold_dir": "synthetic_verify",
        "source_name": "verify_patch_999",
        "source_type": "synthetic",
        "ground_truth_organ": "Breast",
        "organ": {
            "predicted_organ": "Breast",
            "confidence_score": 0.8812,
            "raw_confidence": 0.9410,
            "top_3_predictions": [
                {"organ": "Breast", "confidence": 0.8812},
                {"organ": "Lung", "confidence": 0.0710},
                {"organ": "Colon", "confidence": 0.0241},
            ],
            "calibration_status": {
                "temperature": 2.0,
                "fitted": False,
                "method": "temperature_scaling",
            },
            "model_source": "EfficientNet-B0",
            "ground_truth_organ": "Breast",
            "resolution": "256x256 px",
            "stain_type": "H&E",
        },
        "cells": {
            "total_cell_count": len(centroids),
            "class_counts": counts,
            "composition_percentages": {
                k: round(v / len(centroids) * 100.0, 2) for k, v in counts.items()
            },
            "detection_source": "Synthetic verification",
        },
        "spatial": {
            "graph_topology": {
                "total_nodes": graph_metrics["total_nodes"],
                "total_edges": graph_metrics["total_edges"],
                "avg_clustering_coefficient": graph_metrics["avg_clustering_coefficient"],
                "avg_neighbor_distance_px": graph_metrics["avg_neighbor_distance_px"],
                "cellular_density_per_100x100px": graph_metrics["cellular_density_per_100x100px"],
                "connected_components": graph_metrics["connected_components"],
            },
            "interaction_matrix": graph_metrics["interaction_matrix"],
            "immune_infiltration": graph_metrics["immune_infiltration"],
            "validity_assessment": validity,
            "max_distance_px": 50.0,
            "graph_image_path": graph_metrics["graph_image_path"],
        },
        "literature_evidence": {"status": "no_results", "results": []},
    }
    report = render_deterministic_report(payload)
    audit = evaluate_report_fidelity(report, payload)
    assert audit["passed"], f"Fidelity audit failed: {audit}"
    print(f"   PASS: nodes={graph_metrics['total_nodes']}, fidelity_score={audit['score']:.1f}/100")

    print("\nAll BioCellAgents verification checks passed!")
    return True


def main() -> None:
    args = build_arg_parser().parse_args()
    test_pipeline(offline=args.offline, preferred_device=args.device)


if __name__ == "__main__":
    main()
