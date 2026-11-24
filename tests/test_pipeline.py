"""Offline unit tests for src.pipeline canonical v2.0 schema and custom image pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.pipeline import (
    assemble_canonical_payload,
    build_precomputed_payload_from_image,
    load_patch,
    serialize_precomputed_payload,
)


def _mock_organ_result() -> dict:
    return {
        "predicted_organ": "Breast",
        "confidence_score": 0.9123,
        "confidence": 0.9123,
        "raw_confidence": 0.9501,
        "top_3_predictions": [
            {"organ": "Breast", "confidence": 0.9123},
            {"organ": "Lung", "confidence": 0.0501},
            {"organ": "Colon", "confidence": 0.0204},
        ],
        "top3": [
            {"organ": "Breast", "confidence": 0.9123},
            {"organ": "Lung", "confidence": 0.0501},
            {"organ": "Colon", "confidence": 0.0204},
        ],
        "temperature": 1.659,
        "calibration_status": {
            "temperature": 1.659,
            "fitted": True,
            "method": "temperature_scaling",
            "ece": 0.028,
        },
        "model_source": "EfficientNet-B0",
        "ground_truth_organ": "Breast",
        "resolution": "256x256 px",
        "stain_type": "H&E",
        "time": 0.05,
    }


def _mock_cell_result() -> dict:
    return {
        "total_cell_count": 35,
        "total_cells": 35,
        "class_counts": {
            "Neoplastic": 15,
            "Inflammatory": 8,
            "Connective": 6,
            "Dead": 2,
            "Epithelial": 4,
        },
        "composition_percentages": {
            "Neoplastic": 42.86,
            "Inflammatory": 22.86,
            "Connective": 17.14,
            "Dead": 5.71,
            "Epithelial": 11.43,
        },
        "detection_source": "DINOv2 ViT-S/14",
        "centroids": [{"id": 1, "x": 10.0, "y": 20.0, "class": "Neoplastic"}],
        "detected": [{"cell_id": 1}],
        "graph_topology": {
            "total_nodes": 35,
            "total_edges": 80,
            "avg_clustering_coefficient": 0.35,
            "graph_density": 0.1345,
            "avg_neighbor_distance_px": 24.0,
            "avg_neighbor_distance_um": 6.0,
            "cellular_density_per_100x100px": 5.34,
            "connected_components": 1,
        },
        "interaction_matrix": {"Neoplastic-Inflammatory": 18},
        "immune_infiltration": {
            "mean_immune_to_tumor_dist_px": 28.0,
            "closest_immune_to_tumor_dist_px": 9.2,
            "mean_immune_to_tumor_dist_um": 7.0,
            "closest_immune_to_tumor_dist_um": 2.3,
            "inflammatory_count": 8,
            "neoplastic_count": 15,
        },
        "tme_phenotype": "Immune-Infiltrated",
        "tme_rationale": "Within 40 px margin",
        "validity": {"is_valid": True, "warnings": [], "recommendations": []},
        "max_distance_px": 50.0,
        "graph_image_path": "reports/spatial_graph_patch_0.png",
        "mask": np.ones((256, 256), dtype=np.uint8),
        "time": 0.12,
    }


class TestPipelineCanonicalSchema:
    def test_assemble_and_serialize_strips_mask_and_matches_schema(self):
        payload = assemble_canonical_payload(
            organ_result=_mock_organ_result(),
            cell_result=_mock_cell_result(),
            patch_index=0,
            fold_dir="data/raw/fold3",
            source_name="fold3_patch_0",
            ground_truth_organ="Breast",
        )
        assert payload["schema_version"] == "2.0"
        assert "organ" in payload and "cells" in payload and "spatial" in payload
        assert "deterministic_tables_markdown" in payload
        assert "mask" not in payload["cells"]

        # Inject a raw mask explicitly and verify serializer strips it cleanly
        payload_with_mask = dict(payload)
        payload_with_mask["cells"] = dict(
            payload["cells"], mask=np.zeros((256, 256), dtype=np.uint8)
        )
        raw_json = serialize_precomputed_payload(payload_with_mask)
        decoded = json.loads(raw_json)
        assert "mask" not in decoded["cells"]
        assert decoded["organ"]["confidence_score"] == 0.9123
        assert decoded["spatial"]["graph_topology"]["total_nodes"] == 35

    def test_build_precomputed_payload_from_image_offline(self):
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        with (
            patch("src.pipeline.run_organ_classification", return_value=_mock_organ_result()),
            patch("src.pipeline.run_cell_analysis", return_value=_mock_cell_result()),
        ):
            payload = build_precomputed_payload_from_image(
                dummy_img,
                source_name="custom_slide.png",
                fetch_literature=False,
            )
        assert payload["source_type"] == "custom_image"
        assert payload["source_name"] == "custom_slide.png"
        assert payload["organ"]["predicted_organ"] == "Breast"
        assert payload["cells"]["total_cell_count"] == 35

    def test_load_patch_clamps_index_and_normalizes_float_images(self, tmp_path: Path):
        imgs = np.full((2, 32, 32, 3), 0.5, dtype=np.float32)
        tissues = np.array(["Breast", "Lung"], dtype=object)
        np.save(tmp_path / "images.npy", imgs)
        np.save(tmp_path / "tissues.npy", tissues)

        img, label = load_patch(str(tmp_path), patch_index=99)
        assert img.dtype == np.uint8
        assert img.shape == (32, 32, 3)
        assert label == "Lung"

        with pytest.raises(FileNotFoundError):
            load_patch(str(tmp_path / "missing"), patch_index=0)
