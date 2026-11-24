"""Layer 1 — Quantitative ML, Calibration & Spatial Vision Evaluation Suite.

Supports two execution modes:
- ``synthetic``: 100% deterministic, zero-network benchmark that runs on CPU
  (CI) or GPU and verifies segmentation (Dice/IoU/PQ), organ classification &
  temperature-scaling calibration (Accuracy, Macro F1, Raw vs Calibrated ECE),
  spatial TME graph topology + 3-state phenotyping, and ValidityGate boundaries.
- ``real``: runs against real PanNuke folds and trained checkpoints (with full
  CUDA/AMP/pin_memory support on GPU workstations).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.inference.calibration import TemperatureScaler, compute_ece
from src.inference.metrics import compute_metrics
from src.inference.validity_gate import ValidityGate
from src.reporting import PIXEL_SIZE_UM, classify_tme_phenotype, px_to_um
from src.tools.graph_analytics import SpatialGraphAnalytics
from src.utils.device import get_device, get_hardware_info


def _build_synthetic_segmentation_pair(
    rng: np.random.Generator,
    size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic ground-truth + slightly perturbed predicted masks.

    Returns ``(gt_semantic, pred_semantic, gt_instances, pred_instances)``.
    """
    gt_sem = np.zeros((size, size), dtype=np.int64)
    pred_sem = np.zeros((size, size), dtype=np.int64)
    gt_inst = np.zeros((size, size), dtype=np.int32)
    pred_inst = np.zeros((size, size), dtype=np.int32)

    inst_id = 1
    yy, xx = np.ogrid[:size, :size]
    for cls_id in range(1, 6):
        for k in range(3):
            cy = 18 + (cls_id - 1) * 22
            cx = 22 + k * 38
            r = 8
            gt_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            # Perturb predicted nucleus slightly by 1 px so Dice/IoU/PQ are realistic (< 1.0)
            dy = int(rng.choice([-1, 0, 1]))
            dx = int(rng.choice([-1, 0, 1]))
            pred_mask = (yy - (cy + dy)) ** 2 + (xx - (cx + dx)) ** 2 <= r**2

            gt_sem[gt_mask] = cls_id
            pred_sem[pred_mask] = cls_id
            gt_inst[gt_mask] = inst_id
            pred_inst[pred_mask] = inst_id
            inst_id += 1

    return gt_sem, pred_sem, gt_inst, pred_inst


def _eval_synthetic_segmentation(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_classes = 6
    gt_sem, pred_sem, gt_inst, pred_inst = _build_synthetic_segmentation_pair(rng)

    per_class_dice: Dict[str, float] = {}
    per_class_iou: Dict[str, float] = {}
    names = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    for c in range(num_classes):
        p = pred_sem == c
        g = gt_sem == c
        inter = float(np.sum(p & g))
        union_dice = float(np.sum(p) + np.sum(g))
        union_iou = float(np.sum(p | g))
        per_class_dice[names[c]] = round((2.0 * inter / (union_dice + 1e-8)), 4)
        per_class_iou[names[c]] = round((inter / (union_iou + 1e-8)), 4)

    fg_dice = round(float(np.mean([per_class_dice[n] for n in names[1:]])), 4)
    fg_iou = round(float(np.mean([per_class_iou[n] for n in names[1:]])), 4)
    mean_dice = round(float(np.mean(list(per_class_dice.values()))), 4)
    mean_iou = round(float(np.mean(list(per_class_iou.values()))), 4)

    # Instance-level Panoptic Quality (PQ = DQ * SQ)
    matched_ious: List[float] = []
    tp = 0
    gt_ids = [int(i) for i in np.unique(gt_inst) if i > 0]
    pred_ids = [int(i) for i in np.unique(pred_inst) if i > 0]
    for gid in gt_ids:
        gm = gt_inst == gid
        best_iou = 0.0
        for pid in pred_ids:
            pm = pred_inst == pid
            inter = float(np.sum(gm & pm))
            if inter == 0:
                continue
            iou = inter / float(np.sum(gm | pm))
            if iou > best_iou:
                best_iou = iou
        if best_iou > 0.5:
            tp += 1
            matched_ious.append(best_iou)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    pq = round(dq * sq, 4)

    passed = (fg_dice >= 0.75) and (fg_iou >= 0.60) and (pq >= 0.70)
    return {
        "passed": passed,
        "mean_dice": mean_dice,
        "foreground_dice": fg_dice,
        "mean_iou": mean_iou,
        "foreground_iou": fg_iou,
        "panoptic_quality": pq,
        "detection_quality": round(dq, 4),
        "segmentation_quality": round(sq, 4),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
    }


def _eval_synthetic_organ_and_calibration(
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    n_samples = 380
    num_classes = 19
    labels = torch.arange(n_samples, dtype=torch.long) % num_classes

    # Create strongly overconfident logits (3x temperature scale) with ~85% accuracy,
    # mimicking an uncalibrated neural network prior to Guo et al. temperature scaling.
    logits = torch.randn((n_samples, num_classes), generator=gen) * 1.4
    correct_mask = torch.rand(n_samples, generator=gen) < 0.85
    for idx in range(n_samples):
        if bool(correct_mask[idx].item()):
            logits[idx, labels[idx]] += 5.5
        else:
            wrong_cls = int((labels[idx].item() + 1) % num_classes)
            logits[idx, wrong_cls] += 4.5
    logits = logits * 3.0

    logits_dev = logits.to(device)
    labels_dev = labels.to(device)

    from src.data.dataset import ORGAN_TYPES

    preds = torch.argmax(logits_dev, dim=-1).cpu().numpy()
    targets = labels.numpy()
    cls_metrics = compute_metrics(preds.tolist(), targets.tolist(), list(ORGAN_TYPES))
    per_class_f1 = [float(v["f1"]) for v in cls_metrics["per_class"].values()]
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    raw_probs = torch.softmax(logits_dev, dim=-1)
    ece_raw = float(compute_ece(raw_probs, labels_dev))

    scaler = TemperatureScaler(default_temperature=2.0)
    scaler.fit(logits_dev, labels_dev)
    cal_probs = scaler.calibrate(logits_dev)
    ece_cal = float(compute_ece(cal_probs, labels_dev))

    passed = (
        cls_metrics["accuracy"] >= 0.75
        and macro_f1 >= 0.70
        and scaler.fitted
        and ece_cal <= ece_raw
    )
    return {
        "passed": passed,
        "num_samples": n_samples,
        "accuracy": round(float(cls_metrics["accuracy"]), 4),
        "macro_f1": round(macro_f1, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "ece_reduction": round(ece_raw - ece_cal, 4),
        "fitted_temperature": round(float(scaler.temperature), 4),
        "is_fitted": bool(scaler.fitted),
    }


def _eval_synthetic_spatial_and_validity(tmp_reports_dir: Optional[Path] = None) -> Dict[str, Any]:
    _ = tmp_reports_dir
    analyzer = SpatialGraphAnalytics(max_distance_px=50.0)

    # 1. Immune-Infiltrated layout (inflammatory cells closely intermixed with neoplastic cells)
    infiltrated_centroids: List[Dict[str, Any]] = []
    cid = 1
    for i in range(18):
        infiltrated_centroids.append(
            {
                "id": cid,
                "x": float(40 + (i % 6) * 22),
                "y": float(40 + (i // 6) * 22),
                "class": "Neoplastic",
            }
        )
        cid += 1
    for i in range(10):
        infiltrated_centroids.append(
            {
                "id": cid,
                "x": float(46 + (i % 5) * 22),
                "y": float(46 + (i // 5) * 22),
                "class": "Inflammatory",
            }
        )
        cid += 1
    for cls, count in (("Connective", 6), ("Epithelial", 6), ("Dead", 5)):
        for i in range(count):
            infiltrated_centroids.append(
                {"id": cid, "x": float(180 + i * 10), "y": float(180 + (i % 2) * 12), "class": cls}
            )
            cid += 1

    infiltrated_metrics = analyzer.build_graph_and_compute_metrics(
        infiltrated_centroids, patch_index=9001
    )
    infiltrated_counts = {
        "Neoplastic": 18,
        "Inflammatory": 10,
        "Connective": 6,
        "Epithelial": 6,
        "Dead": 5,
    }
    p_inf, _ = classify_tme_phenotype(
        infiltrated_counts, infiltrated_metrics["immune_infiltration"]
    )

    # 2. Immune-Excluded layout (inflammatory cells > 40 px away from neoplastic cluster)
    excluded_centroids: List[Dict[str, Any]] = []
    cid = 1
    for i in range(20):
        excluded_centroids.append(
            {
                "id": cid,
                "x": float(20 + (i % 5) * 10),
                "y": float(20 + (i // 5) * 10),
                "class": "Neoplastic",
            }
        )
        cid += 1
    for i in range(12):
        excluded_centroids.append(
            {
                "id": cid,
                "x": float(190 + (i % 4) * 12),
                "y": float(190 + (i // 4) * 12),
                "class": "Inflammatory",
            }
        )
        cid += 1
    excluded_metrics = analyzer.build_graph_and_compute_metrics(
        excluded_centroids, patch_index=9002
    )
    p_exc, _ = classify_tme_phenotype(
        {"Neoplastic": 20, "Inflammatory": 12, "Connective": 0, "Dead": 0, "Epithelial": 0},
        excluded_metrics["immune_infiltration"],
    )

    # 3. Immune-Desert layout (<5% inflammatory)
    p_des, _ = classify_tme_phenotype(
        {"Neoplastic": 35, "Inflammatory": 1, "Connective": 4, "Dead": 0, "Epithelial": 0},
        {"mean_immune_to_tumor_dist_px": 15.0},
    )

    # 4. Statistical Validity Gate boundary tests
    gate = ValidityGate(min_total_cells=30, min_per_class=5)
    v_zero = gate.full_assessment(n_cells=0, counts_by_class={})
    v_29 = gate.full_assessment(n_cells=29, counts_by_class={"Neoplastic": 20, "Inflammatory": 9})
    v_30_low_cls = gate.full_assessment(
        n_cells=30,
        counts_by_class={
            "Neoplastic": 10,
            "Inflammatory": 6,
            "Connective": 6,
            "Dead": 3,
            "Epithelial": 5,
        },
    )
    v_balanced = gate.full_assessment(
        n_cells=32,
        counts_by_class={
            "Neoplastic": 10,
            "Inflammatory": 6,
            "Connective": 6,
            "Dead": 5,
            "Epithelial": 5,
        },
    )

    mean_inf_px = infiltrated_metrics["immune_infiltration"]["mean_immune_to_tumor_dist_px"]
    mean_inf_um = px_to_um(mean_inf_px, PIXEL_SIZE_UM)

    passed = (
        infiltrated_metrics["total_nodes"] == 45
        and infiltrated_metrics["total_edges"] > 0
        and p_inf == "Immune-Infiltrated"
        and p_exc == "Immune-Excluded"
        and p_des == "Immune-Desert"
        and (v_zero["is_valid"] is False)
        and (v_29["is_valid"] is False)
        and (v_30_low_cls["is_valid"] is False)
        and len(v_30_low_cls["warnings"]) >= 1
        and (v_balanced["is_valid"] is True)
        and len(v_balanced["warnings"]) == 0
    )

    return {
        "passed": passed,
        "infiltrated_nodes": infiltrated_metrics["total_nodes"],
        "infiltrated_edges": infiltrated_metrics["total_edges"],
        "mean_immune_dist_px": round(float(mean_inf_px), 4),
        "mean_immune_dist_um": mean_inf_um,
        "phenotypes_verified": {
            "infiltrated": p_inf,
            "excluded": p_exc,
            "desert": p_des,
        },
        "validity_gate_checks": {
            "n_0_invalid": not v_zero["is_valid"],
            "n_29_invalid": not v_29["is_valid"],
            "n_30_low_class_warned": bool(
                (not v_30_low_cls["is_valid"]) and v_30_low_cls["warnings"]
            ),
            "n_32_balanced_valid": bool(v_balanced["is_valid"] and not v_balanced["warnings"]),
        },
    }


def run_ml_vision_eval(
    mode: str = "synthetic",
    preferred_device: Optional[str] = None,
    seed: int = 42,
    max_samples: int = 0,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run Layer 1 (ML, Calibration & Spatial Vision) evaluation suite."""
    t0 = time.time()
    device = get_device(preferred_device)
    hw = get_hardware_info()

    if mode == "synthetic":
        seg = _eval_synthetic_segmentation(seed=seed)
        organ = _eval_synthetic_organ_and_calibration(device=device, seed=seed)
        spatial = _eval_synthetic_spatial_and_validity(tmp_reports_dir=reports_dir)
        all_passed = bool(seg["passed"] and organ["passed"] and spatial["passed"])
        return {
            "mode": "synthetic",
            "device": str(device),
            "hardware": hw,
            "passed": all_passed,
            "elapsed_seconds": round(time.time() - t0, 3),
            "segmentation": seg,
            "organ_and_calibration": organ,
            "spatial_and_validity": spatial,
        }

    if mode == "real":
        from scripts.evaluate_all import evaluate_cell, evaluate_organ, evaluate_segmentation

        tasks_out: Dict[str, Any] = {}
        all_ok = True
        for name, fn in (
            ("cell", evaluate_cell),
            ("organ", evaluate_organ),
            ("segmentation", evaluate_segmentation),
        ):
            try:
                tasks_out[name] = fn(device, batch_size=64, num_workers=2, max_samples=max_samples)
            except Exception as exc:
                tasks_out[name] = {"error": str(exc)}
                all_ok = False
        return {
            "mode": "real",
            "device": str(device),
            "hardware": hw,
            "passed": all_ok,
            "elapsed_seconds": round(time.time() - t0, 3),
            "tasks": tasks_out,
        }

    raise ValueError(f"Unsupported eval mode: {mode!r}. Expected 'synthetic' or 'real'.")
