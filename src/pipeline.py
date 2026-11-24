"""
BioCellAgents — Shared Pre-Computation Pipeline
================================================
Deterministic quantitative vision, calibration, spatial graph, and validity
pre-computation pipeline shared across the CLI (``main.py``), the Streamlit
dashboard (``biocell_app.py``), and the 3-agent ``PathologyCrew``.

Produces a canonical, schema-aligned ``precomputed_results`` payload (v2.0)
with bulky 2D NumPy masks stripped prior to JSON serialization so LLM agents
receive exact, compact, verbatim-ready quantitative data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.dataset import CELL_TYPES, ORGAN_TYPES
from src.data.transforms import clean_organ_label
from src.inference.calibration import TemperatureScaler
from src.inference.validity_gate import ValidityGate
from src.models.cell_detector import TrainedCellDetector
from src.models.organ_classifier import OrganClassifier
from src.reporting import (
    PIXEL_SIZE_UM,
    build_deterministic_tables_bundle,
    classify_tme_phenotype,
    px_to_um,
)
from src.tools.graph_analytics import SpatialGraphAnalytics
from src.utils.device import get_device, get_hardware_info, load_checkpoint_safe

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGAN_CKPT = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt"
CALIBRATION_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "calibration.json"
SEGMENTER_CKPT = "checkpoints/segmentation_vits14/best_segmentation_model.pt"
CLASSIFIER_CKPT = "checkpoints/cell_vits14/best_model.pt"

# --- Module-level model caches (avoid re-loading on repeated calls) ---
_organ_model_cache: Optional[Tuple[OrganClassifier, torch.device]] = None
_cell_detector_cache: Optional[TrainedCellDetector] = None


def get_organ_classifier(device_name: Optional[str] = None) -> Tuple[OrganClassifier, torch.device]:
    """Load (or return cached) organ classifier model on GPU/CPU."""
    global _organ_model_cache
    if _organ_model_cache is not None and device_name is None:
        return _organ_model_cache
    device = get_device(device_name)
    model = OrganClassifier(num_classes=len(ORGAN_TYPES)).to(device)
    if ORGAN_CKPT.exists():
        state_dict = load_checkpoint_safe(ORGAN_CKPT, map_location=device, extract_state_dict=True)
        model.load_state_dict(state_dict)
    model.eval()
    if device_name is None:
        _organ_model_cache = (model, device)
    return model, device


def get_cell_detector() -> TrainedCellDetector:
    """Load (or return cached) cell detector."""
    global _cell_detector_cache
    if _cell_detector_cache is not None:
        return _cell_detector_cache
    detector = TrainedCellDetector(
        model_size="vits14",
        segmenter_ckpt=SEGMENTER_CKPT,
        classifier_ckpt=CLASSIFIER_CKPT,
    )
    _cell_detector_cache = detector
    return detector


def _normalize_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any 2D/3D/4-channel image array to contiguous HxWx3 uint8 RGB."""
    raw = np.asarray(arr)
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    elif raw.ndim == 3 and raw.shape[2] == 1:
        raw = np.repeat(raw, 3, axis=2)

    if raw.dtype != np.uint8:
        if raw.size > 0 and float(np.nanmax(raw)) <= 1.0:
            raw = np.clip(np.nan_to_num(raw, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
        else:
            raw = np.clip(np.nan_to_num(raw, nan=0.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(raw)


# --- Data loading ---
def load_patch(fold_dir: str, patch_index: int) -> Tuple[np.ndarray, str]:
    """Load H&E image patch + tissue label from PanNuke ``.npy`` files.

    Returns:
        ``(image, tissue_label)`` where ``image`` is uint8 HxWx3 RGB.
    """
    images_path = Path(fold_dir) / "images.npy"
    tissues_path = Path(fold_dir) / "tissues.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"images.npy not found in {fold_dir}")
    images = np.load(images_path, mmap_mode="r")
    n_patches = len(images)
    if n_patches == 0:
        raise ValueError(f"images.npy is empty in {fold_dir}")
    patch_index = max(0, min(int(patch_index), n_patches - 1))
    raw_img = _normalize_rgb_uint8(np.array(images[patch_index]))
    tissue = (
        clean_organ_label(np.load(tissues_path, allow_pickle=True)[patch_index])
        if tissues_path.exists()
        else "Unknown"
    )
    return raw_img, tissue


def load_image_file(image_path: str | Path) -> np.ndarray:
    """Load a standalone H&E image (PNG/JPG/TIFF) as uint8 HxWx3 RGB."""
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as img:
        return _normalize_rgb_uint8(np.array(img.convert("RGB")))


# --- Inference ---
def run_organ_classification(
    image: np.ndarray,
    ground_truth_organ: str = "Unknown",
) -> Dict[str, Any]:
    """Run calibrated 19-organ classification on a single H&E patch."""
    t0 = time.time()
    rgb = _normalize_rgb_uint8(image)
    model, device = get_organ_classifier()
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    raw_probs = torch.softmax(logits[0], dim=-1)
    scaler = TemperatureScaler.load(CALIBRATION_PATH)
    calibrated = scaler.calibrate(logits[0])
    k = min(3, len(ORGAN_TYPES))
    top3_probs, top3_idx = torch.topk(calibrated, k=k)

    predicted_organ = ORGAN_TYPES[int(top3_idx[0].item())]
    cal_conf = round(float(top3_probs[0].cpu().item()), 4)
    raw_conf = round(float(raw_probs[int(top3_idx[0].item())].cpu().item()), 4)
    top3_list = [
        {"organ": ORGAN_TYPES[int(i.item())], "confidence": round(float(p.cpu().item()), 4)}
        for p, i in zip(top3_probs, top3_idx)
    ]
    h, w = rgb.shape[:2]
    model_source = (
        f"EfficientNet-B0 ({ORGAN_CKPT.name})"
        if ORGAN_CKPT.exists()
        else "EfficientNet-B0 (untrained/fallback weights)"
    )

    return {
        "predicted_organ": predicted_organ,
        "confidence_score": cal_conf,
        "confidence": cal_conf,  # backward-compatible alias
        "raw_confidence": raw_conf,
        "top_3_predictions": top3_list,
        "top3": top3_list,  # backward-compatible alias
        "temperature": round(float(scaler.temperature), 4),
        "calibration_status": {
            "temperature": round(float(scaler.temperature), 4),
            "fitted": bool(getattr(scaler, "is_fitted", False)),
            "method": "temperature_scaling",
            "ece": getattr(scaler, "ece", None),
        },
        "model_source": model_source,
        "ground_truth_organ": ground_truth_organ,
        "resolution": f"{w}x{h} px",
        "stain_type": "H&E",
        "time": round(time.time() - t0, 4),
    }


def run_cell_analysis(
    image: np.ndarray,
    max_distance_px: float = 50.0,
    patch_index: int = 0,
) -> Dict[str, Any]:
    """Run cell detection + 5-class classification + spatial graph + validity audit."""
    t0 = time.time()
    rgb = _normalize_rgb_uint8(image)
    detector = get_cell_detector()
    detected, mask = detector.detect_and_classify(rgb)
    centroids = detector.to_centroids(detected)
    raw_counts = detector.class_counts(detected)

    # Guarantee all 5 canonical PanNuke classes are present
    counts: Dict[str, int] = {cls: int(raw_counts.get(cls, 0)) for cls in CELL_TYPES}
    total_cells = len(centroids)
    composition_percentages: Dict[str, float] = {
        cls: round((counts[cls] / total_cells) * 100.0, 2) if total_cells > 0 else 0.0
        for cls in CELL_TYPES
    }

    # Spatial graph
    analyzer = SpatialGraphAnalytics(max_distance_px=max_distance_px)
    metrics = analyzer.build_graph_and_compute_metrics(
        centroids, patch_index=patch_index, he_image=rgb
    )

    # Validity gate
    gate = ValidityGate(min_total_cells=30, min_per_class=5)
    validity = gate.full_assessment(n_cells=total_cells, counts_by_class=counts)

    # Enrich immune infiltration with micrometers
    immune_raw = dict(metrics.get("immune_infiltration") or {})
    mean_px = immune_raw.get("mean_immune_to_tumor_dist_px")
    min_px = immune_raw.get("closest_immune_to_tumor_dist_px")
    immune_enriched: Dict[str, Any] = {
        **immune_raw,
        "mean_immune_to_tumor_dist_um": px_to_um(mean_px, PIXEL_SIZE_UM),
        "closest_immune_to_tumor_dist_um": px_to_um(min_px, PIXEL_SIZE_UM),
    }
    tme_phenotype, tme_rationale = classify_tme_phenotype(
        counts, immune_enriched, pixel_size_um=PIXEL_SIZE_UM
    )

    avg_neighbor_px = metrics.get("avg_neighbor_distance_px")
    total_nodes = int(metrics.get("total_nodes", total_cells))
    total_edges = int(metrics.get("total_edges", 0))
    graph_density = (
        round((2.0 * total_edges) / (total_nodes * (total_nodes - 1)), 4)
        if total_nodes > 1
        else 0.0
    )

    graph_topology: Dict[str, Any] = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_clustering_coefficient": round(
            float(metrics.get("avg_clustering_coefficient", 0.0)), 4
        ),
        "graph_density": graph_density,
        "avg_neighbor_distance_px": (
            round(float(avg_neighbor_px), 4) if avg_neighbor_px is not None else 0.0
        ),
        "avg_neighbor_distance_um": px_to_um(avg_neighbor_px, PIXEL_SIZE_UM),
        "cellular_density_per_100x100px": round(
            float(metrics.get("cellular_density_per_100x100px", 0.0)), 4
        ),
        "connected_components": int(metrics.get("connected_components", 0)),
    }

    return {
        "total_cell_count": total_cells,
        "total_cells": total_cells,  # backward-compatible alias
        "class_counts": counts,
        "composition_percentages": composition_percentages,
        "detection_source": getattr(detector, "detection_source", "DINOv2 ViT-S/14"),
        "centroids": centroids,
        "detected": detected,
        "graph_metrics": metrics,
        "graph_topology": graph_topology,
        "interaction_matrix": metrics.get("interaction_matrix", {}),
        "immune_infiltration": immune_enriched,
        "tme_phenotype": tme_phenotype,
        "tme_rationale": tme_rationale,
        "validity": validity,
        "max_distance_px": max_distance_px,
        "graph_image_path": metrics.get("graph_image_path", "N/A"),
        "mask": mask,
        "time": round(time.time() - t0, 4),
    }


def _retrieve_literature_evidence(
    organ: str,
    class_counts: Dict[str, int],
    tme_phenotype: str,
    top_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Pre-retrieve PubMed + Europe PMC citations (best-effort, non-fatal)."""
    try:
        from src.tools.pubmed_search_tool import LiteratureSearchTool

        lit_tool = LiteratureSearchTool()
        dominant_cell = max(class_counts, key=lambda k: class_counts[k]) if class_counts else None
        terms: List[str] = []
        if organ and organ.lower() != "unknown":
            terms.append(organ.replace("_", " "))
        if tme_phenotype:
            terms.append(tme_phenotype.replace("-", " ").lower())
        if dominant_cell:
            terms.append(f"{dominant_cell.lower()}-rich")
        terms += ["tumor microenvironment", "spatial topology", "immune infiltration"]
        query = " ".join(terms)
        lit_result = lit_tool._run(query=query, top_k=top_k)
        if lit_result.get("status") == "success" and lit_result.get("results"):
            print(f"[pipeline] Pre-retrieved {len(lit_result['results'])} literature citations")
            return lit_result
        print(f"[pipeline] Literature pre-retrieval: {lit_result.get('message', 'no results')}")
        return None
    except Exception as exc:
        print(f"[pipeline] Literature pre-retrieval failed (non-fatal): {exc}")
        return None


def assemble_canonical_payload(
    organ_result: Dict[str, Any],
    cell_result: Dict[str, Any],
    *,
    patch_index: int = 0,
    fold_dir: Optional[str] = None,
    source_name: str = "patch_0",
    source_type: str = "pannuke_fold",
    ground_truth_organ: str = "Unknown",
    literature_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical v2.0 precomputed payload (without bulky 2D masks)."""
    organ_clean = dict(organ_result)
    organ_clean.setdefault("ground_truth_organ", ground_truth_organ)

    cells_clean: Dict[str, Any] = {
        "total_cell_count": cell_result.get("total_cell_count", cell_result.get("total_cells", 0)),
        "total_cells": cell_result.get("total_cells", cell_result.get("total_cell_count", 0)),
        "class_counts": cell_result.get("class_counts", {}),
        "composition_percentages": cell_result.get("composition_percentages", {}),
        "detection_source": cell_result.get("detection_source", "DINOv2 ViT-S/14"),
    }

    spatial_clean: Dict[str, Any] = {
        "graph_topology": cell_result.get("graph_topology", {}),
        "interaction_matrix": cell_result.get("interaction_matrix", {}),
        "immune_infiltration": cell_result.get("immune_infiltration", {}),
        "tme_phenotype": cell_result.get("tme_phenotype", "Unknown"),
        "tme_rationale": cell_result.get("tme_rationale", ""),
        "validity_assessment": cell_result.get("validity", {}),
        "max_distance_px": cell_result.get("max_distance_px", 50.0),
        "graph_image_path": cell_result.get("graph_image_path", "N/A"),
        "pixel_size_um": PIXEL_SIZE_UM,
    }

    payload: Dict[str, Any] = {
        "schema_version": "2.0",
        "patch_index": patch_index,
        "fold_dir": fold_dir,
        "source_name": source_name,
        "source_type": source_type,
        "ground_truth_organ": ground_truth_organ,
        "hardware": get_hardware_info(),
        "organ": organ_clean,
        "cells": cells_clean,
        "spatial": spatial_clean,
    }
    if literature_evidence is not None:
        payload["literature_evidence"] = literature_evidence
    else:
        payload["literature_evidence"] = {"status": "not_retrieved", "results": []}

    # Pre-render exact Markdown tables so agents never manually re-type numeric tables
    payload["deterministic_tables_markdown"] = build_deterministic_tables_bundle(payload)
    return payload


def serialize_precomputed_payload(payload: Dict[str, Any]) -> str:
    """Serialize a canonical payload to compact JSON, stripping any raw numpy arrays."""

    def _strip_bulky(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _strip_bulky(v)
                for k, v in obj.items()
                if k not in {"mask", "centroids", "detected"}
            }
        if isinstance(obj, list):
            return [_strip_bulky(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    return json.dumps(_strip_bulky(payload), indent=2, ensure_ascii=False, default=str)


# --- Full pipeline ---
def build_precomputed_payload_from_image(
    image: np.ndarray,
    source_name: str = "uploaded_patch",
    ground_truth_organ: str = "Unknown",
    *,
    fold_dir: Optional[str] = None,
    patch_index: int = 0,
    source_type: str = "custom_image",
    max_distance_px: float = 50.0,
    fetch_literature: bool = True,
) -> Dict[str, Any]:
    """Run full pre-computation on an in-memory H&E RGB image and return canonical dict."""
    print(f"[pipeline] Running organ classification on {source_name}...")
    organ_result = run_organ_classification(image, ground_truth_organ=ground_truth_organ)
    print(
        f"[pipeline]   -> {organ_result['predicted_organ']} "
        f"({organ_result['confidence_score']:.1%})"
    )

    print("[pipeline] Running cell detection + spatial analysis...")
    cell_result = run_cell_analysis(image, max_distance_px=max_distance_px, patch_index=patch_index)
    print(f"[pipeline]   -> {cell_result['total_cell_count']} cells detected")

    lit_evidence = None
    if fetch_literature:
        lit_evidence = _retrieve_literature_evidence(
            organ=str(organ_result.get("predicted_organ", "unknown")),
            class_counts=cell_result.get("class_counts", {}),
            tme_phenotype=str(cell_result.get("tme_phenotype", "")),
        )

    return assemble_canonical_payload(
        organ_result=organ_result,
        cell_result=cell_result,
        patch_index=patch_index,
        fold_dir=fold_dir,
        source_name=source_name,
        source_type=source_type,
        ground_truth_organ=ground_truth_organ,
        literature_evidence=lit_evidence,
    )


def build_precomputed_results_from_image(
    image: np.ndarray,
    source_name: str = "uploaded_patch",
    ground_truth_organ: str = "Unknown",
    *,
    max_distance_px: float = 50.0,
    fetch_literature: bool = True,
) -> str:
    """Run full pre-computation on an in-memory H&E image and return compact JSON string."""
    payload = build_precomputed_payload_from_image(
        image=image,
        source_name=source_name,
        ground_truth_organ=ground_truth_organ,
        max_distance_px=max_distance_px,
        fetch_literature=fetch_literature,
    )
    serialized = serialize_precomputed_payload(payload)
    print(f"[pipeline] Precomputed results: {len(serialized)} chars")
    return serialized


def build_precomputed_payload(
    fold_dir: str,
    patch_index: int,
    *,
    max_distance_px: float = 50.0,
    fetch_literature: bool = True,
) -> Dict[str, Any]:
    """Load a PanNuke fold patch and run full pre-computation, returning canonical dict."""
    print(f"[pipeline] Loading patch {patch_index} from {fold_dir}...")
    image, tissue = load_patch(fold_dir, patch_index)
    print(f"[pipeline] Image shape: {image.shape}, tissue: {tissue}")
    return build_precomputed_payload_from_image(
        image=image,
        source_name=f"{Path(fold_dir).name}_patch_{patch_index}",
        ground_truth_organ=tissue,
        fold_dir=fold_dir,
        patch_index=patch_index,
        source_type="pannuke_fold",
        max_distance_px=max_distance_px,
        fetch_literature=fetch_literature,
    )


def build_precomputed_results(
    fold_dir: str,
    patch_index: int,
    *,
    max_distance_px: float = 50.0,
    fetch_literature: bool = True,
) -> str:
    """Run the full pre-computation pipeline on a PanNuke fold patch and return JSON string."""
    payload = build_precomputed_payload(
        fold_dir=fold_dir,
        patch_index=patch_index,
        max_distance_px=max_distance_px,
        fetch_literature=fetch_literature,
    )
    serialized = serialize_precomputed_payload(payload)
    print(f"[pipeline] Precomputed results: {len(serialized)} chars")
    return serialized
