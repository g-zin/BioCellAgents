"""
BioCellAgents — Shared Pre-Computation Pipeline
================================================
Extracts the inference logic from biocell_app.py into a reusable module
so that both the Streamlit app and the CLI entry point (main.py) can
produce the `precomputed_results` payload expected by the CrewAI tasks.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ORGAN_TYPES
from src.models.organ_classifier import OrganClassifier
from src.inference.calibration import TemperatureScaler
from src.models.cell_detector import TrainedCellDetector
from src.tools.graph_analytics import SpatialGraphAnalytics
from src.inference.validity_gate import ValidityGate

# --- Constants (mirrors biocell_app.py) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGAN_CKPT = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt"
CALIBRATION_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "calibration.json"
SEGMENTER_CKPT = "checkpoints/segmentation_vits14/best_segmentation_model.pt"
CLASSIFIER_CKPT = "checkpoints/cell_vits14/best_model.pt"

# --- Module-level model caches (avoid re-loading on repeated calls) ---
_organ_model_cache: tuple | None = None
_cell_detector_cache: TrainedCellDetector | None = None


def get_organ_classifier() -> tuple[OrganClassifier, torch.device]:
    """Load (or return cached) organ classifier model."""
    global _organ_model_cache
    if _organ_model_cache is not None:
        return _organ_model_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrganClassifier(num_classes=len(ORGAN_TYPES)).to(device)
    if ORGAN_CKPT.exists():
        ckpt = torch.load(ORGAN_CKPT, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
    model.eval()
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


# --- Data loading ---
def load_patch(fold_dir: str, patch_index: int) -> tuple[np.ndarray, str]:
    """Load H&E image patch + tissue label from .npy files.

    Returns:
        (image, tissue_label) — image is uint8 HxWx3.
    """
    images_path = Path(fold_dir) / "images.npy"
    tissues_path = Path(fold_dir) / "tissues.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"images.npy not found in {fold_dir}")
    images = np.load(images_path, mmap_mode="r")
    n_patches = len(images)
    if patch_index >= n_patches:
        patch_index = 0
    raw_img = np.array(images[patch_index])
    if raw_img.max() <= 1.0:
        raw_img = (raw_img * 255).astype(np.uint8)
    else:
        raw_img = raw_img.astype(np.uint8)
    tissue = str(np.load(tissues_path, allow_pickle=True)[patch_index]) if tissues_path.exists() else "Unknown"
    return raw_img, tissue


# --- Inference ---
def run_organ_classification(image: np.ndarray) -> dict:
    """Run organ classification on a single H&E patch."""
    t0 = time.time()
    model, device = get_organ_classifier()
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    scaler = TemperatureScaler.load(CALIBRATION_PATH)
    calibrated = scaler.calibrate(logits[0])
    top3_probs, top3_idx = torch.topk(calibrated, k=3)
    return {
        "predicted_organ": ORGAN_TYPES[top3_idx[0].item()],
        "confidence": float(top3_probs[0].cpu().numpy()),
        "top3": [
            {"organ": ORGAN_TYPES[i.item()], "confidence": round(float(p.cpu().numpy()), 4)}
            for p, i in zip(top3_probs, top3_idx)
        ],
        "temperature": scaler.temperature,
        "time": time.time() - t0,
    }


def run_cell_analysis(image: np.ndarray) -> dict:
    """Run cell detection + classification + spatial graph analysis."""
    t0 = time.time()
    detector = get_cell_detector()
    detected, mask = detector.detect_and_classify(image)
    centroids = detector.to_centroids(detected)
    counts = detector.class_counts(detected)

    # Spatial graph
    analyzer = SpatialGraphAnalytics(max_distance_px=50.0)
    metrics = analyzer.build_graph_and_compute_metrics(centroids, patch_index=0, he_image=image)

    # Validity gate
    gate = ValidityGate(min_total_cells=30, min_per_class=5)
    validity = gate.full_assessment(n_cells=len(centroids), counts_by_class=counts)

    return {
        "total_cells": len(centroids),
        "class_counts": counts,
        "centroids": centroids,
        "detected": detected,
        "graph_metrics": metrics,
        "validity": validity,
        "mask": mask,
        "time": time.time() - t0,
    }


# --- Full pipeline ---
def build_precomputed_results(fold_dir: str, patch_index: int) -> str:
    """Run the full pre-computation pipeline and return a JSON string.

    This is the value passed as `precomputed_results` in the CrewAI kickoff inputs.
    """
    print(f"[pipeline] Loading patch {patch_index} from {fold_dir}...")
    image, tissue = load_patch(fold_dir, patch_index)
    print(f"[pipeline] Image shape: {image.shape}, tissue: {tissue}")

    print("[pipeline] Running organ classification...")
    organ_result = run_organ_classification(image)
    print(f"[pipeline]   → {organ_result['predicted_organ']} ({organ_result['confidence']:.1%})")

    print("[pipeline] Running cell detection + spatial analysis...")
    cell_result = run_cell_analysis(image)
    print(f"[pipeline]   → {cell_result['total_cells']} cells detected")

    # Assemble the results dict (same structure as biocell_app.py)
    results = {
        "patch_index": patch_index,
        "fold_dir": fold_dir,
        "ground_truth_organ": tissue,
        "organ": organ_result,
        "cells": cell_result,
    }

    # -- Pre-retrieval: Inject literature evidence (PubMed + Europe PMC) --
    try:
        from src.tools.pubmed_search_tool import LiteratureSearchTool
        lit_tool = LiteratureSearchTool()
        organ = organ_result.get("predicted_organ", "unknown")
        class_counts = cell_result.get("class_counts", {}) or {}
        dominant_cell = max(class_counts, key=class_counts.get) if class_counts else None

        terms = []
        if organ and organ.lower() != "unknown":
            terms.append(organ.replace("_", " "))
        if dominant_cell:
            terms.append(f"{dominant_cell.lower()}-rich")
        terms += ["tumor microenvironment", "spatial topology", "immune infiltration", "classification"]
        query = " ".join(terms)
        lit_result = lit_tool._run(query=query, top_k=5)
        if lit_result.get("status") == "success" and lit_result.get("results"):
            results["literature_evidence"] = lit_result
            print(f"[pipeline] Pre-retrieved {len(lit_result['results'])} literature citations")
        else:
            print(f"[pipeline] Literature pre-retrieval: {lit_result.get('message', 'no results')}")
    except Exception as e:
        print(f"[pipeline] Literature pre-retrieval failed (non-fatal): {e}")

    # Serialize to JSON (same as biocell_app.py)
    precomputed = json.dumps(results, indent=2, default=str)
    print(f"[pipeline] Precomputed results: {len(precomputed)} chars")
    return precomputed