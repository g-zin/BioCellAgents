"""
BioCellAgents — Clinical Dark Precision Digital Pathology & Spatial Biology Platform
====================================================================================
Unified Streamlit application combining:
- PanNuke multi-fold browser (fold1 / fold2 / fold3) + Custom H&E image upload
- Live GPU/CPU hardware telemetry badge (CUDA model, VRAM, PyTorch version)
- Calibrated 19-organ EfficientNet-B0 + DINOv2 5-class nuclei segmentation & typing
- Interactive NetworkX/Plotly spatial TME graph, pairwise heatmap & infiltration depth
- Deterministic 11-section RUO report engine + Genuinely Tool-Using 3-Agent CrewAI mode
- Integrated Dual-Layer Eval Benchmark & Live Report Fidelity Grader tab

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.dashboard.charts import build_cell_pie as _dashboard_cell_pie
from src.dashboard.charts import build_interaction_count_heatmap
from src.dashboard.charts import build_organ_bar as _dashboard_organ_bar
from src.dashboard.charts import build_spatial_scatter as _dashboard_spatial_scatter
from src.dashboard.styles import CELL_COLORS, get_custom_css, get_palette
from src.data.dataset import CELL_TYPES, ORGAN_TYPES
from src.data.transforms import clean_organ_label
from src.evals.report_fidelity_eval import evaluate_report_fidelity
from src.evals.runner import run_full_eval_suite, run_report_fidelity_suite
from src.pipeline import (
    CALIBRATION_PATH,
    CLASSIFIER_CKPT,
    ORGAN_CKPT,
    SEGMENTER_CKPT,
    assemble_canonical_payload,
)
from src.pipeline import get_cell_detector as _pipeline_cell_detector
from src.pipeline import get_organ_classifier as _pipeline_organ_classifier
from src.pipeline import run_cell_analysis as _pipeline_cell_analysis
from src.pipeline import run_organ_classification as _pipeline_organ_classification
from src.pipeline import serialize_precomputed_payload
from src.reporting import (
    PIXEL_SIZE_UM,
    export_report_pdf,
    format_distance,
    render_deterministic_report,
)
from src.tools.graph_analytics import load_he_patch
from src.utils.device import get_hardware_info

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep backward-compatible module exports
_ = (
    Image,
    load_he_patch,
    ORGAN_TYPES,
    ORGAN_CKPT,
    CALIBRATION_PATH,
    SEGMENTER_CKPT,
    CLASSIFIER_CKPT,
    CELL_COLORS,
)

FOLD_DIRS: Dict[str, str] = {
    "Fold 3 (PanNuke) - Held-Out Test": "data/raw/fold3",
    "Fold 2 (PanNuke) - Validation": "data/raw/fold2",
    "Fold 1 (PanNuke) - Train": "data/raw/fold1",
}

EVAL_SUMMARY_PATH = PROJECT_ROOT / "results" / "evals" / "eval_summary.json"
CSS = get_custom_css()


def _configure_page() -> None:
    try:
        st.set_page_config(
            page_title="BioCellAgents — Digital Pathology & Spatial Biology",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def discover_available_folds() -> Dict[str, str]:
    """Return available PanNuke folds on disk, falling back to default map if none exist."""
    found = {
        label: rel
        for label, rel in FOLD_DIRS.items()
        if (PROJECT_ROOT / rel / "images.npy").exists()
    }
    return found if found else dict(FOLD_DIRS)


@st.cache_resource(show_spinner=False)
def get_organ_classifier():
    return _pipeline_organ_classifier()


@st.cache_resource(show_spinner=False)
def get_cell_detector():
    return _pipeline_cell_detector()


@st.cache_data(show_spinner=False)
def load_patch_data(
    fold_dir: str, patch_index: int
) -> Tuple[Optional[np.ndarray], Optional[str], int]:
    """Load an H&E patch + cleaned tissue label from a PanNuke fold directory."""
    images_path = Path(fold_dir) / "images.npy"
    tissues_path = Path(fold_dir) / "tissues.npy"
    if not images_path.exists():
        return None, None, 0
    images = np.load(images_path, mmap_mode="r")
    n_patches = int(len(images))
    if n_patches == 0:
        return None, None, 0
    idx = max(0, min(int(patch_index), n_patches - 1))
    raw_img = np.array(images[idx])
    if raw_img.dtype != np.uint8:
        if raw_img.size > 0 and float(raw_img.max()) <= 1.0:
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
        else:
            raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)
    tissue = (
        clean_organ_label(np.load(tissues_path, allow_pickle=True)[idx])
        if tissues_path.exists()
        else "Unknown"
    )
    return raw_img, tissue, n_patches


def run_organ_classification(image: np.ndarray) -> Dict[str, Any]:
    return _pipeline_organ_classification(image)


def run_cell_analysis(
    image: np.ndarray,
    max_distance_px: float = 50.0,
) -> Dict[str, Any]:
    return _pipeline_cell_analysis(image, max_distance_px=max_distance_px)


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.size > 0 and float(arr.max()) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def build_segmentation_overlay(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    palette: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Blend a semantic/binary nuclei mask over an H&E RGB patch."""
    rgb = _ensure_rgb_uint8(image).copy()
    if mask is None or mask.size == 0:
        return rgb
    colors = palette or CELL_COLORS
    overlay = rgb.copy().astype(np.float32)

    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    if mask.shape[:2] != rgb.shape[:2]:
        pil_mask = Image.fromarray(mask.astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST
        )
        mask_resized = np.array(pil_mask)
    else:
        mask_resized = mask

    for cls_idx, cls_name in enumerate(CELL_TYPES, start=1):
        region = mask_resized == cls_idx
        if np.any(region):
            c_rgb = np.array(_hex_to_rgb(colors.get(cls_name, "#4FC3F7")), dtype=np.float32)
            overlay[region] = 0.55 * overlay[region] + 0.45 * c_rgb

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _patch_litellm_for_groq() -> None:
    import litellm
    import litellm.main

    litellm.drop_params = True
    litellm.cache = None

    if getattr(litellm.main.completion, "_biocell_patched", False):
        return

    _orig_completion = litellm.main.completion

    def _strip(msgs: Any) -> Any:
        if not isinstance(msgs, list):
            return msgs
        return [
            {k: v for k, v in m.items() if k != "cache_breakpoint"} if isinstance(m, dict) else m
            for m in msgs
        ]

    def _patched_completion(*args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = _strip(kwargs["messages"])
        elif args and isinstance(args[0], list):
            args = (_strip(args[0]),) + args[1:]
        return _orig_completion(*args, **kwargs)

    setattr(_patched_completion, "_biocell_patched", True)
    litellm.main.completion = _patched_completion
    litellm.completion = _patched_completion


def run_crewai_pipeline(
    fold_dir: str,
    patch_index: int,
    results: Dict[str, Any],
    on_task_complete: Optional[Callable[[Any], None]] = None,
) -> str:
    """Execute the tool-using 3-agent PathologyCrew on a precomputed results dict."""
    _patch_litellm_for_groq()
    from src.crew import PathologyCrew
    from src.tools.pubmed_search_tool import LiteratureSearchTool

    enriched = dict(results)
    if "literature_evidence" not in enriched or not enriched["literature_evidence"].get("results"):
        try:
            organ = str((enriched.get("organ") or {}).get("predicted_organ", "unknown"))
            lit_res = LiteratureSearchTool()._run(
                query=f"{organ.replace('_', ' ')} tumor microenvironment spatial biology",
                top_k=5,
            )
            if lit_res.get("results"):
                enriched["literature_evidence"] = lit_res
        except Exception:
            pass

    precomputed = serialize_precomputed_payload(enriched)
    crew_obj = PathologyCrew().crew()
    if on_task_complete is not None and hasattr(crew_obj, "task_callback"):
        crew_obj.task_callback = on_task_complete

    output = crew_obj.kickoff(
        inputs={
            "fold_dir": fold_dir,
            "patch_index": patch_index,
            "precomputed_results": precomputed,
        }
    )
    return str(output.raw) if hasattr(output, "raw") else str(output)


def run_chat_query(
    question: str,
    results: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    """Answer a user pathology question grounded strictly in precomputed quantitative results."""
    import litellm

    litellm.drop_params = True
    model = os.getenv("MODEL", "groq/qwen/qwen3.8-27b")
    context_json = serialize_precomputed_payload(results)[:6000]
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a computational pathology research assistant (Research Use Only). "
                "Answer strictly from the provided quantitative JSON results without fabricating "
                "numbers or issuing clinical diagnoses.\n\n"
                f"QUANTITATIVE CONTEXT:\n{context_json}"
            ),
        }
    ]
    for turn in history[-6:]:
        if "role" in turn and "content" in turn:
            messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
    messages.append({"role": "user", "content": question})

    resp = litellm.completion(model=model, messages=messages)
    content = resp.choices[0].message.content
    return "" if content is None else str(content)


# --- Unified Chart Exports (backward-compatible signatures) ---
def build_cell_pie(
    counts: Dict[str, int],
    colorblind_safe: bool = False,
) -> go.Figure:
    return _dashboard_cell_pie(counts, palette=get_palette(colorblind_safe))


def build_organ_bar(top3: List[Dict[str, Any]]) -> go.Figure:
    return _dashboard_organ_bar(top3)


def build_spatial_scatter(
    centroids: List[Dict[str, Any]],
    colorblind_safe: bool = False,
) -> go.Figure:
    return _dashboard_spatial_scatter(centroids, palette=get_palette(colorblind_safe))


# --- UI Components ---
def render_metric_card(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def render_hardware_badge() -> None:
    hw = get_hardware_info()
    if hw.get("cuda_available"):
        vram = hw.get("vram_total_gb")
        vram_txt = f" · {vram:.1f} GB VRAM" if vram else ""
        gpu_label = hw.get("gpu_name") or hw.get("device", "CUDA GPU")
        text = f"⚡ GPU: {gpu_label}{vram_txt}"
        cls = "hw-badge hw-badge-gpu"
    else:
        text = f"🖥️ CPU Mode · PyTorch {hw.get('torch_version', 'N/A')}"
        cls = "hw-badge hw-badge-cpu"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def render_agent_pipeline(completed: bool = False, deterministic: bool = False) -> None:
    steps = [
        ("1. Python Pre-Compute", "Vision + Calibration + Spatial (0% drift)"),
        (
            "2. Adaptive Literature" if not deterministic else "2. Offline Citations",
            "PubMed/Europe PMC Tool" if not deterministic else "Deterministic Formatter",
        ),
        (
            "3. Clinical Synthesis" if not deterministic else "3. Deterministic 11-Sec",
            "RUO Narrative + Tables" if not deterministic else "Instant Zero-LLM Report",
        ),
        (
            "4. Fidelity Auditor",
            "Programmatic Verifier Tool (100-pt Rubric)",
        ),
    ]
    cols = st.columns(len(steps))
    step_cls = "pipeline-step pipeline-step-done" if completed else "pipeline-step"
    for col, (title, sub) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="{step_cls}">'
                f'<div class="pipeline-step-title">{title}</div>'
                f'<div class="pipeline-step-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )


def _render_eval_tab() -> None:
    st.markdown("### Dual-Layer Eval Benchmark & Report Fidelity Suite")
    st.caption(
        "Layer 1 validates Segmentation (Dice/IoU/PQ), Organ F1 + Temperature Calibration ECE, "
        "Spatial TME Phenotyping, and ValidityGate boundaries. Layer 2 grades 11-section report "
        "fidelity across 6 criteria plus 5 adversarial hallucination probes."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run Fast Report Fidelity Eval (Layer 2)", key="btn_eval_l2"):
            with st.spinner("Running Layer 2 Report Fidelity Suite..."):
                l2 = run_report_fidelity_suite()
                st.session_state["eval_summary"] = {
                    "passed": l2["passed"],
                    "metadata": {"mode": "fidelity_only", "device": "cpu"},
                    "layer1_ml_vision": None,
                    "layer2_report_fidelity": l2,
                }
    with col_b:
        if st.button("Run Full Synthetic Dual-Layer Eval (L1 + L2)", key="btn_eval_full"):
            with st.spinner("Running Full Dual-Layer Synthetic Eval Suite..."):
                st.session_state["eval_summary"] = run_full_eval_suite(
                    mode="synthetic",
                    output_dir=EVAL_SUMMARY_PATH.parent,
                )

    summary: Optional[Dict[str, Any]] = st.session_state.get("eval_summary")
    if summary is None and EVAL_SUMMARY_PATH.exists():
        try:
            summary = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    if summary is None:
        st.info(
            "No evaluation scorecard loaded yet. Click a button above or run "
            "`python scripts/run_evals.py --mode synthetic` from your terminal."
        )
        return

    gate_cls = "status-pass" if summary.get("passed") else "status-fail"
    gate_lbl = "PASS" if summary.get("passed") else "FAIL"
    st.markdown(
        f'<span class="status-pill {gate_cls}">OVERALL EVAL GATE: {gate_lbl}</span>',
        unsafe_allow_html=True,
    )

    l1 = summary.get("layer1_ml_vision")
    if isinstance(l1, dict) and l1.get("mode") == "synthetic":
        seg = l1.get("segmentation", {})
        org = l1.get("organ_and_calibration", {})
        spa = l1.get("spatial_and_validity", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                "Fg Dice / PQ",
                f"{seg.get('foreground_dice', 0):.3f}",
                f"PQ: {seg.get('panoptic_quality', 0):.3f}",
            )
        with c2:
            render_metric_card(
                "Organ Macro F1",
                f"{org.get('macro_f1', 0):.3f}",
                f"Acc: {org.get('accuracy', 0):.3f}",
            )
        with c3:
            render_metric_card(
                "Calibration ECE",
                f"{org.get('ece_calibrated', 0):.4f}",
                f"Raw: {org.get('ece_raw', 0):.4f} · τ={org.get('fitted_temperature', 0):.2f}",
            )
        with c4:
            render_metric_card(
                "Spatial & Gate",
                "PASS" if spa.get("passed") else "FAIL",
                f"{spa.get('infiltrated_nodes', 0)} nodes / {spa.get('infiltrated_edges', 0)} edges",
            )

    l2 = summary.get("layer2_report_fidelity")
    if isinstance(l2, dict):
        c_lit = l2.get("compliant_with_literature", {})
        c_nolit = l2.get("compliant_no_literature", {})
        catch_pct = float(l2.get("adversarial_catch_rate", 0.0)) * 100.0
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card(
                "Compliant (PubMed)", f"{c_lit.get('score', 0):.1f}/100", "11-Section RUO"
            )
        with m2:
            render_metric_card(
                "Compliant (Offline)", f"{c_nolit.get('score', 0):.1f}/100", "Fallback Grounded"
            )
        with m3:
            render_metric_card(
                "Adversarial Catch Rate", f"{catch_pct:.0f}%", "5 Hallucination Probes"
            )


def main() -> None:
    _configure_page()

    st.markdown(
        '<div class="biocell-hero">'
        '<div class="biocell-hero-title">BioCellAgents — Computational Pathology & Spatial Biology</div>'
        '<div class="biocell-hero-sub">'
        "Calibrated Multi-Organ Vision · 5-Class PanNuke Nuclei Topology · "
        "Tool-Using 3-Agent & Deterministic RUO Reporting"
        "</div></div>",
        unsafe_allow_html=True,
    )

    available_folds = discover_available_folds()

    # --- Sidebar Control Center ---
    with st.sidebar:
        st.markdown("### Runtime Hardware")
        render_hardware_badge()
        st.divider()

        st.markdown("### Input Source")
        input_mode = st.selectbox(
            "Patch Source",
            ["PanNuke Fold", "Custom H&E Image Upload"],
            key="input_mode",
        )

        selected_fold_dir = ""
        patch_idx = 0
        uploaded_file = None

        if input_mode == "PanNuke Fold":
            fold_label = st.selectbox("Dataset Fold", list(available_folds.keys()))
            selected_fold_dir = str(PROJECT_ROOT / available_folds[fold_label])
            patch_idx = int(
                st.number_input("Patch Index", min_value=0, max_value=10000, value=0, step=1)
            )
        else:
            file_uploader = getattr(st, "file_uploader", None)
            if callable(file_uploader):
                uploaded_file = file_uploader(
                    "Upload H&E Patch (PNG / JPG / TIF)",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                )

        st.divider()
        st.markdown("### Visualization & Pipeline")
        colorblind_safe = st.checkbox("Okabe-Ito Colorblind Palette", value=False)
        skip_ai = st.checkbox("Skip AI Report (faster)", value=False)
        deterministic_mode = st.checkbox(
            "Deterministic Offline Report (Zero-LLM)",
            value=not bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")),
        )
        slider_fn = getattr(st, "slider", None)
        max_dist_px = (
            float(slider_fn("Spatial Radius (px)", 15.0, 120.0, 50.0, 5.0))
            if callable(slider_fn)
            else 50.0
        )

        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    # --- Load Active Tissue Patch ---
    raw_image: Optional[np.ndarray] = None
    gt_tissue: str = "Unknown"
    source_name: str = f"patch_{patch_idx}"
    source_type: str = "pannuke_fold"

    if input_mode == "Custom H&E Image Upload":
        source_type = "custom_image"
        if uploaded_file is not None:
            raw_image = _ensure_rgb_uint8(np.array(Image.open(uploaded_file).convert("RGB")))
            source_name = getattr(uploaded_file, "name", "uploaded_patch.png")
    else:
        raw_image, loaded_tissue, n_total = load_patch_data(selected_fold_dir, patch_idx)
        gt_tissue = loaded_tissue or "Unknown"
        source_name = f"{Path(selected_fold_dir).name}_patch_{patch_idx}"
        if raw_image is None and not run_clicked:
            st.info(
                f"PanNuke data not found at `{selected_fold_dir}`. Place PanNuke folds under "
                "`data/raw/fold1..3/` or switch **Patch Source** to **Custom H&E Image Upload**."
            )

    # --- Execute Analysis Pipeline ---
    if run_clicked and raw_image is not None:
        t_start = time.time()
        with st.spinner("Running calibrated organ classification & spatial nuclei analysis..."):
            organ_res = run_organ_classification(raw_image)
            try:
                cell_res = run_cell_analysis(raw_image, max_distance_px=max_dist_px)
            except TypeError:
                # Supports unit-test monkeypatches with 1-argument signature
                cell_res = run_cell_analysis(raw_image)

        payload = assemble_canonical_payload(
            organ_result=organ_res,
            cell_result=cell_res,
            patch_index=patch_idx,
            fold_dir=selected_fold_dir if source_type == "pannuke_fold" else None,
            source_name=source_name,
            source_type=source_type,
            ground_truth_organ=gt_tissue,
        )

        report_md = ""
        fidelity_audit: Optional[Dict[str, Any]] = None
        if not skip_ai:
            if deterministic_mode:
                report_md = render_deterministic_report(payload)
            else:
                with st.spinner("Running 3-Agent Tool-Using PathologyCrew..."):
                    report_md = run_crewai_pipeline(
                        selected_fold_dir or ".",
                        patch_idx,
                        payload,
                    )
            fidelity_audit = evaluate_report_fidelity(report_md, payload)

        st.session_state["analysis"] = {
            "image": _ensure_rgb_uint8(raw_image),
            "gt_tissue": gt_tissue,
            "organ": organ_res,
            "cells": cell_res,
            "payload": payload,
            "report_md": report_md,
            "fidelity_audit": fidelity_audit,
            "deterministic": deterministic_mode,
            "elapsed": round(time.time() - t_start, 2),
        }

    analysis = st.session_state.get("analysis")
    palette = get_palette(colorblind_safe)

    # --- Render Workspace (with clean fallback if st.tabs is absent in minimal mocks) ---
    tabs_fn = getattr(st, "tabs", None)
    if callable(tabs_fn):
        tab_overview, tab_spatial, tab_report, tab_eval = tabs_fn(
            [
                "🔬 Pathology Overview",
                "🕸️ Spatial TME & Graph",
                "📋 RUO Report & Q&A",
                "📊 Eval Benchmark & Fidelity",
            ]
        )
    else:
        from contextlib import nullcontext

        tab_overview = tab_spatial = tab_report = tab_eval = nullcontext()

    with tab_overview:
        if analysis is None:
            st.caption(
                "Select a PanNuke patch or upload an H&E image, then click **Run Analysis**."
            )
        else:
            organ_res = analysis["organ"]
            cell_res = analysis["cells"]
            gm = cell_res.get("graph_metrics") or cell_res.get("graph_topology") or {}
            validity = cell_res.get("validity") or {}

            pred_organ = organ_res.get("predicted_organ", "Unknown")
            conf_val = float(organ_res.get("confidence_score", organ_res.get("confidence", 0.0)))
            tau = float(organ_res.get("temperature", 2.0))
            total_cells = int(cell_res.get("total_cell_count", cell_res.get("total_cells", 0)))
            tme_label = cell_res.get("tme_phenotype", "Analyzed")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_metric_card(
                    "Predicted Organ", str(pred_organ), f"GT: {analysis['gt_tissue']}"
                )
            with k2:
                render_metric_card("Calibrated Conf.", f"{conf_val:.1%}", f"τ = {tau:.3f}")
            with k3:
                render_metric_card("Detected Nuclei", str(total_cells), "5-Class PanNuke")
            with k4:
                val_status = "VALID" if validity.get("is_valid") else "LOW-N WARN"
                render_metric_card("TME / Validity", str(tme_label), val_status)

            if validity and not validity.get("is_valid", True):
                for warn in validity.get("warnings", []):
                    st.warning(str(warn))

            c_img1, c_img2 = st.columns(2)
            with c_img1:
                st.image(
                    analysis["image"], caption="Input H&E Tissue Patch", use_container_width=True
                )
            with c_img2:
                overlay = build_segmentation_overlay(
                    analysis["image"], cell_res.get("mask"), palette=palette
                )
                st.image(
                    overlay, caption="5-Class Nuclei Segmentation Overlay", use_container_width=True
                )

            ch1, ch2 = st.columns(2)
            with ch1:
                top3 = organ_res.get("top_3_predictions") or organ_res.get("top3") or []
                st.plotly_chart(build_organ_bar(top3), use_container_width=True)
            with ch2:
                counts = cell_res.get("class_counts") or {}
                st.plotly_chart(
                    build_cell_pie(counts, colorblind_safe=colorblind_safe),
                    use_container_width=True,
                )

    with tab_spatial:
        if analysis is None:
            st.caption("Run an analysis to inspect spatial proximity graphs and TME topology.")
        else:
            cell_res = analysis["cells"]
            gm = cell_res.get("graph_metrics") or cell_res.get("graph_topology") or {}
            nodes = int(gm.get("total_nodes", 0))
            edges = int(gm.get("total_edges", 0))
            avg_deg = (2.0 * edges / nodes) if nodes > 0 else 0.0
            comps = int(gm.get("connected_components", 0))

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                render_metric_card("Graph Nodes", str(nodes), "Cells (V)")
            with s2:
                render_metric_card("Graph Edges", str(edges), f"r <= {max_dist_px:.0f} px")
            with s3:
                render_metric_card("Avg Degree", f"{avg_deg:.2f}", "2E / V")
            with s4:
                render_metric_card("Components", str(comps), "Connected Subgraphs")

            immune = cell_res.get("immune_infiltration") or gm.get("immune_infiltration") or {}
            if immune:
                mean_px = immune.get("mean_immune_to_tumor_dist_px")
                min_px = immune.get("closest_immune_to_tumor_dist_px")
                st.caption(
                    f"Immune-to-Tumor Depth — Mean: **{format_distance(mean_px, PIXEL_SIZE_UM)}** "
                    f"| Closest: **{format_distance(min_px, PIXEL_SIZE_UM)}**"
                )

            g1, g2 = st.columns(2)
            with g1:
                centroids = cell_res.get("centroids") or []
                st.plotly_chart(
                    build_spatial_scatter(centroids, colorblind_safe=colorblind_safe),
                    use_container_width=True,
                )
            with g2:
                inter_mat = cell_res.get("interaction_matrix") or gm.get("interaction_matrix") or {}
                hmap = build_interaction_count_heatmap(inter_mat)
                if hmap is not None:
                    st.plotly_chart(hmap, use_container_width=True)
                else:
                    st.info("No pairwise cell-cell proximity edges detected within radius.")

    with tab_report:
        render_agent_pipeline(
            completed=bool(analysis and analysis.get("report_md")),
            deterministic=(
                bool(analysis.get("deterministic", deterministic_mode))
                if analysis
                else deterministic_mode
            ),
        )
        if analysis is None:
            st.caption("Run an analysis to generate a verified 11-section RUO pathology report.")
        else:
            report_md = analysis.get("report_md", "")
            audit = analysis.get("fidelity_audit")
            if audit:
                pill_cls = "status-pass" if audit.get("passed") else "status-warn"
                st.markdown(
                    f'<span class="status-pill {pill_cls}">'
                    f"FIDELITY AUDIT: {audit['score']:.1f}/100 "
                    f"({'PASS' if audit['passed'] else 'REVIEW'})</span>",
                    unsafe_allow_html=True,
                )
            if report_md:
                st.markdown(report_md)
                dl_fn = getattr(st, "download_button", None)
                if callable(dl_fn):
                    d1, d2 = st.columns(2)
                    with d1:
                        dl_fn(
                            "Download Report (.md)",
                            data=report_md,
                            file_name="biocell_pathology_report.md",
                            mime="text/markdown",
                        )
                    with d2:
                        try:
                            pdf_bytes = export_report_pdf(report_md)
                            dl_fn(
                                "Download Report (.pdf)",
                                data=pdf_bytes,
                                file_name="biocell_pathology_report.pdf",
                                mime="application/pdf",
                            )
                        except Exception:
                            pass
            else:
                st.info(
                    "AI/Deterministic report generation was skipped. Uncheck 'Skip AI Report' to generate."
                )

            # Grounded Q&A
            chat_history = st.session_state.setdefault("chat_history", [])
            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            prompt = st.chat_input(
                "Ask a question about this tissue patch's quantitative metrics..."
            )
            if prompt:
                chat_history.append({"role": "user", "content": prompt})
                try:
                    answer = run_chat_query(prompt, analysis["payload"], chat_history[:-1])
                except Exception as exc:
                    answer = f"Chat unavailable (configure GROQ_API_KEY / OPENAI_API_KEY): {exc}"
                chat_history.append({"role": "assistant", "content": answer})

    with tab_eval:
        if callable(tabs_fn):
            _render_eval_tab()


if __name__ == "__main__":
    main()
