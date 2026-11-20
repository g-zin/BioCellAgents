"""
BioCell - Enterprise Digital Pathology & Spatial Biology Platform
==================================================================
Real PanNuke data + DINOv2 models + CrewAI multi-agent pipeline.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CELL_TYPES, ORGAN_TYPES  # noqa: E402
from src.data.transforms import clean_organ_label  # noqa: E402
from src.inference.calibration import TemperatureScaler  # noqa: E402
from src.inference.validity_gate import ValidityGate  # noqa: E402
from src.models.cell_detector import TrainedCellDetector  # noqa: E402
from src.models.organ_classifier import OrganClassifier  # noqa: E402
from src.tools.graph_analytics import SpatialGraphAnalytics, load_he_patch  # noqa: E402

# Keep load_he_patch and Image referenced for parity with module exports
_ = (Image, load_he_patch)

FOLD_DIRS: Dict[str, str] = {
    "Fold 3 (PanNuke) - Test": "data/raw/fold3",
}

ORGAN_CKPT = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "best_model.pt"
CALIBRATION_PATH = PROJECT_ROOT / "checkpoints" / "organ_vits14" / "calibration.json"
SEGMENTER_CKPT = "checkpoints/segmentation_vits14/best_segmentation_model.pt"
CLASSIFIER_CKPT = "checkpoints/cell_vits14/best_model.pt"

CELL_COLORS: Dict[str, str] = {
    "Neoplastic": "#e63946",
    "Inflammatory": "#457b9d",
    "Connective": "#2a9d8f",
    "Dead": "#2b2d42",
    "Epithelial": "#f4a261",
    "Unknown": "#8d99ae",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.stApp { background: #0f1419; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.bio-header {
    border-bottom: 3px solid #1a73e8; padding: 20px 24px; margin-bottom: 24px;
    background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
    border-radius: 12px; border: 1px solid #2a3a4a;
}
.bio-title { font-size: 1.7rem; font-weight: 800; color: #e8eaed; margin-bottom: 4px; letter-spacing: -0.03em; }
.bio-subtitle { font-size: 0.9rem; color: #9aa0a6; font-weight: 500; }
.bio-meta { display: flex; gap: 20px; margin-top: 12px; font-size: 0.78rem; color: #5f6368; }
.bio-meta span { white-space: nowrap; }
.metric-card {
    background: #1a2332; border: 1px solid #2a3a4a; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 12px;
}
.metric-label {
    font-size: 0.72rem; color: #9aa0a6; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 6px;
}
.metric-value { font-size: 1.5rem; font-weight: 700; color: #e8eaed; }
.metric-sub { font-size: 0.78rem; color: #5f6368; margin-top: 4px; }
.finding { padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 0.85rem; }
.finding-info { background: #1a2332; border-left: 3px solid #1a73e8; color: #e8eaed; }
.finding-warn { background: #1a2332; border-left: 3px solid #f4a261; color: #e8eaed; }
.finding-error { background: #1a2332; border-left: 3px solid #e63946; color: #e8eaed; }
.finding-success { background: #1a2332; border-left: 3px solid #2a9d8f; color: #e8eaed; }
.agent-row {
    display: flex; align-items: center; gap: 12px; padding: 12px 16px;
    background: #1a2332; border: 1px solid #2a3a4a; border-radius: 10px; margin-bottom: 8px;
}
.agent-icon { font-size: 1.4rem; }
.agent-name { font-weight: 600; color: #e8eaed; font-size: 0.9rem; }
.agent-status { font-size: 0.78rem; color: #9aa0a6; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
.badge-blue { background: #1a3a5c; color: #8ab4f8; }
.badge-green { background: #1a3c2a; color: #81c995; }
.badge-amber { background: #3c2a1a; color: #f4a261; }
</style>
"""


def _configure_page() -> None:
    """Apply Streamlit page configuration and dark enterprise CSS."""
    st.set_page_config(
        page_title="BioCellAgents - Computational Pathology",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_organ_classifier() -> Tuple[OrganClassifier, torch.device]:
    """Load the trained organ classifier (EfficientNet-B0)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrganClassifier(num_classes=len(ORGAN_TYPES)).to(device)
    if ORGAN_CKPT.exists():
        ckpt = torch.load(ORGAN_CKPT, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def get_cell_detector() -> TrainedCellDetector:
    """Load the trained cell detector (DINOv2 segmenter + classifier)."""
    detector = TrainedCellDetector(
        model_size="vits14",
        segmenter_ckpt=SEGMENTER_CKPT,
        classifier_ckpt=CLASSIFIER_CKPT,
    )
    return detector


@st.cache_data(show_spinner=False)
def load_patch_data(
    fold_dir: str, patch_index: int
) -> Tuple[Optional[np.ndarray], Optional[str], int]:
    """Load image + tissue label from the .npy files."""
    images_path = Path(fold_dir) / "images.npy"
    tissues_path = Path(fold_dir) / "tissues.npy"
    if not images_path.exists():
        return None, None, 0

    images = np.load(images_path, mmap_mode="r")
    n_patches = len(images)
    if n_patches == 0:
        return None, None, 0
    patch_index = max(0, min(int(patch_index), n_patches - 1))
    raw_img = np.array(images[patch_index])
    if raw_img.dtype != np.uint8:
        if raw_img.size > 0 and raw_img.max() <= 1.0:
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
        else:
            raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)
    tissue = (
        clean_organ_label(np.load(tissues_path, allow_pickle=True)[patch_index])
        if tissues_path.exists()
        else "Unknown"
    )
    return raw_img, tissue, n_patches


def run_organ_classification(image: np.ndarray) -> Dict[str, Any]:
    """Run organ classification on a single H&E patch."""
    t0 = time.time()
    model, device = get_organ_classifier()
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    scaler = TemperatureScaler.load(CALIBRATION_PATH)
    calibrated = scaler.calibrate(logits[0])
    top3_probs, top3_idx = torch.topk(calibrated, k=3)
    return {
        "predicted_organ": ORGAN_TYPES[int(top3_idx[0].item())],
        "confidence": float(top3_probs[0].cpu().numpy()),
        "top3": [
            {
                "organ": ORGAN_TYPES[int(i.item())],
                "confidence": round(float(p.cpu().numpy()), 4),
            }
            for p, i in zip(top3_probs, top3_idx)
        ],
        "temperature": scaler.temperature,
        "time": time.time() - t0,
    }


def run_cell_analysis(image: np.ndarray) -> Dict[str, Any]:
    """Run cell detection + classification + spatial graph analysis."""
    t0 = time.time()
    detector = get_cell_detector()
    detected, mask = detector.detect_and_classify(image)
    centroids = detector.to_centroids(detected)
    counts = detector.class_counts(detected)

    analyzer = SpatialGraphAnalytics(max_distance_px=50.0)
    metrics = analyzer.build_graph_and_compute_metrics(centroids, patch_index=0, he_image=image)

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


def run_crewai_pipeline(
    fold_dir: str,
    patch_index: int,
    results: Dict[str, Any],
    progress_callback: Optional[Callable[[Any], None]] = None,
) -> str:
    """Run the full CrewAI multi-agent pipeline."""
    import litellm
    import litellm.main

    litellm.drop_params = True
    litellm.cache = None

    def _strip_cache_breakpoint(messages: Any) -> Any:
        if not messages:
            return messages
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and "cache_breakpoint" in m:
                m = {k: v for k, v in m.items() if k != "cache_breakpoint"}
            cleaned.append(m)
        return cleaned

    if not getattr(litellm.main.completion, "_biocell_patched", False):
        _orig = litellm.main.completion

        def _patched(*args: Any, **kwargs: Any) -> Any:
            if "messages" in kwargs:
                kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
            elif args and isinstance(args[0], list):
                args = (_strip_cache_breakpoint(args[0]),) + args[1:]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return _orig(*args, **kwargs)
                except Exception as e:
                    if (
                        "rate_limit" in str(type(e).__name__).lower()
                        or "rate limit" in str(e).lower()
                        or "RateLimit" in str(type(e).__name__)
                    ):
                        wait = 15 * (attempt + 1)
                        print(
                            f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait)
                        continue
                    raise
            raise RuntimeError("Groq rate limit: max retries exceeded")

        setattr(_patched, "_biocell_patched", True)
        litellm.main.completion = _patched
        litellm.completion = _patched

    from src.crew import PathologyCrew

    precomputed = json.dumps(results, indent=2, default=str)

    # Pre-retrieve PubMed / Europe PMC literature evidence so the crew stays grounded
    literature_evidence: Optional[Dict[str, Any]] = None
    try:
        from src.tools.pubmed_search_tool import LiteratureSearchTool

        lit_tool = LiteratureSearchTool()
        organ = results.get("organ", {}).get("predicted_organ", "unknown")
        cells = results.get("cells", {}) or {}
        class_counts = cells.get("class_counts", {}) or {}
        dominant_cell = max(class_counts, key=lambda k: class_counts[k]) if class_counts else None

        terms: List[str] = []
        if organ and organ.lower() != "unknown":
            terms.append(organ.replace("_", " "))
        if dominant_cell:
            terms.append(f"{dominant_cell.lower()}-rich")
        terms += [
            "tumor microenvironment",
            "spatial topology",
            "immune infiltration",
            "classification",
        ]
        query = " ".join(terms)
        lit_result = lit_tool._run(query=query, top_k=5)
        if lit_result.get("status") == "success" and lit_result.get("results"):
            literature_evidence = lit_result
            print(
                f"Pre-retrieved {len(lit_result['results'])} literature citations from PubMed/Europe PMC"
            )
        else:
            print(f"Literature pre-retrieval: {lit_result.get('message', 'no results')}")
    except Exception as e:
        print(f"Literature pre-retrieval failed (non-fatal): {e}")

    if literature_evidence:
        results["literature_evidence"] = literature_evidence
        precomputed = json.dumps(results, indent=2, default=str)

    inputs = {
        "fold_dir": fold_dir,
        "patch_index": patch_index,
        "precomputed_results": precomputed,
    }

    crew_instance = PathologyCrew().crew()
    if progress_callback:
        crew_instance.task_callback = progress_callback

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = crew_instance.kickoff(inputs=inputs)
            return str(result.raw) if hasattr(result, "raw") else str(result)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str and "limit" in err_str and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                import time as _time

                _time.sleep(wait)
                continue
            raise
    return ""


def run_chat_query(question: str, results: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    """Answer a free-form question about the current patch using pre-computed results."""
    import litellm

    litellm.drop_params = True

    results_json = json.dumps(results, indent=2, default=str)
    system_prompt = (
        f"You are BioCell's Spatial Query Agent - a computational pathology assistant.\n"
        f"You answer questions about the CURRENT tissue patch ONLY, using the pre-computed data below.\n\n"
        f"PRE-COMPUTED DATA (Patch {results.get('patch_index', '?')}, Fold: {results.get('fold_dir', '?')}):\n"
        f"{results_json}"
        f"\n\nSTRICT RULES:\n"
        f"1. Use ONLY the numbers provided above. NEVER invent, estimate, or extrapolate values.\n"
        f"2. Report distances in BOTH pixels and micrometers (um). Conversion: 0.25 um/px (40x objective).\n"
        f'3. If data is insufficient or a metric is missing, say "N/A" or "insufficient data" - never fabricate.\n'
        f"4. Be concise (2-5 sentences) unless the user explicitly asks for a detailed breakdown.\n"
        f"5. Frame all interpretations as analytical/research findings (RUO), NOT clinical diagnosis.\n"
        f"6. If the validity gate flags the sample as exploratory, mention this caveat.\n"
        f"7. You may reference cell counts, spatial metrics, immune infiltration, and organ classification.\n"
        f"8. If the question is outside the scope of this patch's data, politely state what you CAN answer.\n"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model="groq/openai/gpt-oss-120b",
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str and "limit" in err_str and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"Chat rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Chat: max retries exceeded")


def build_cell_pie(counts: Dict[str, int]) -> go.Figure:
    """Build a donut chart of cell type distribution."""
    labels = [k for k in counts if counts[k] > 0]
    values = [counts[k] for k in labels]
    colors = [CELL_COLORS.get(k, "#8d99ae") for k in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0f1419", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#e8eaed"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def build_organ_bar(top3: List[Dict[str, Any]]) -> go.Figure:
    """Build a horizontal bar chart of top-3 organ predictions."""
    organs = [t.get("organ", "Unknown") for t in top3]
    confs = [t.get("confidence", 0.0) for t in top3]
    fig = go.Figure(
        go.Bar(
            x=confs,
            y=organs,
            orientation="h",
            marker=dict(color=["#1a73e8", "#457b9d", "#2a9d8f"]),
            text=[f"{c:.1%}" for c in confs],
            textposition="outside",
            textfont=dict(size=11, color="#e8eaed"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        margin=dict(l=80, r=40, t=30, b=20),
        height=200,
        xaxis=dict(range=[0, 1], showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def build_spatial_scatter(centroids: List[Dict[str, Any]]) -> go.Figure:
    """Build a scatter plot of cell centroids colored by class."""
    fig = go.Figure()
    for cell_type in CELL_TYPES:
        pts = [c for c in centroids if c.get("class") == cell_type]
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[c.get("x", 0.0) for c in pts],
                y=[c.get("y", 0.0) for c in pts],
                mode="markers",
                name=cell_type,
                marker=dict(size=6, color=CELL_COLORS.get(cell_type, "#8d99ae")),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        margin=dict(l=40, r=20, t=30, b=30),
        height=350,
        xaxis=dict(title="X (px)", showgrid=False),
        yaxis=dict(title="Y (px)", showgrid=False, scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def render_metric_card(label: str, value: str, sub: str = "") -> None:
    """Render a styled metric card in HTML."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_agent_pipeline(completed: bool = False) -> None:
    """Render the 3-agent pipeline visualization."""
    status = "Complete" if completed else "Pending"
    agents = [
        ("", "Pathology CV Analyst", "Organ classification + cell parsing", status),
        ("", "Spatial Bio Analyst", "Cell graph + TME metrics", status),
        ("", "Clinical Report Writer", "Synthesized pathology report", status),
    ]
    for icon, name, desc, status in agents:
        st.markdown(
            f"""
        <div class="agent-row">
            <span class="agent-icon">{icon}</span>
            <div><div class="agent-name">{name}</div>
            <div class="agent-status">{desc} - {status}</div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the BioCellAgents Streamlit dashboard."""
    _configure_page()

    with st.sidebar:
        st.markdown("## Controls")
        fold_label = st.selectbox("Dataset Fold", list(FOLD_DIRS.keys()), index=0)
        fold_dir = FOLD_DIRS[fold_label]

        images_path = Path(fold_dir) / "images.npy"
        n_patches = 0
        if images_path.exists():
            n_patches = len(np.load(images_path, mmap_mode="r"))

        patch_idx = st.number_input(
            "Patch Index", min_value=0, max_value=max(n_patches - 1, 0), value=0, step=1
        )
        st.divider()
        run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")
        skip_ai_report = st.checkbox("Skip AI Report (faster)", value=False)
        st.divider()
        device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.markdown(
            f"""
        <div style="font-size:0.75rem; color:#5f6368;">
            <b>Device:</b> {device_label}<br/>
            <b>Patches:</b> {n_patches}<br/>
            <b>Model:</b> DINOv2 ViT-S/14<br/>
            <b>Organ:</b> EfficientNet-B0
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div class="bio-header">
        <div class="bio-title">BioCellAgents</div>
        <div class="bio-subtitle">Enterprise Digital Pathology & Spatial Biology Platform</div>
        <div class="bio-meta">
            <span>PanNuke Dataset</span>
            <span>DINOv2 ViT-S/14</span>
            <span>CrewAI Multi-Agent</span>
            <span>{datetime.now().strftime('%Y-%m-%d')}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not run_analysis and "results" not in st.session_state:
        st.info("Select a fold and patch, then click **Run Analysis** to begin.")
        render_agent_pipeline()
        return

    image, ground_truth, n_patches = load_patch_data(fold_dir, patch_idx)
    if image is None:
        st.error(
            f"Could not load data from `{fold_dir}`. Ensure the PanNuke dataset is downloaded."
        )
        return

    if run_analysis:
        st.info("Running inference pipeline...")
        results: Dict[str, Any] = {
            "patch_index": patch_idx,
            "fold_dir": fold_dir,
        }
        with st.spinner("Classifying organ..."):
            results["organ"] = run_organ_classification(image)
        with st.spinner("Detecting cells + building spatial graph..."):
            results["cells"] = run_cell_analysis(image)

        st.session_state["results"] = results
        st.session_state["image"] = image
        st.session_state["ground_truth"] = ground_truth
        st.session_state["fold_label"] = fold_label
        st.session_state["patch_idx"] = patch_idx

    if "results" not in st.session_state:
        st.warning("No analysis results yet. Click **Run Analysis** in the sidebar.")
        return

    results = st.session_state["results"]
    image = st.session_state["image"]
    ground_truth = st.session_state["ground_truth"]
    fold_label = st.session_state["fold_label"]
    patch_idx = st.session_state["patch_idx"]

    st.markdown(
        f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <span class="badge badge-blue">PanNuke {fold_label.split()[0]}</span>
        <span class="badge badge-green">Patch #{patch_idx:04d}</span>
        <span class="badge badge-amber">H&E Stain</span>
        <span class="badge badge-blue">{image.shape[1]}x{image.shape[0]} px</span>
        <span style="color:#5f6368; font-size:0.8rem;">
            Ground Truth: <b style="color:#e8eaed;">{ground_truth}</b>
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_img, col_organ = st.columns([1, 1])
    with col_img:
        st.markdown("### H&E Tissue Patch")
        st.image(image, use_container_width=True)

    with col_organ:
        st.markdown("### Organ Classification")
        organ = results["organ"]
        render_metric_card(
            "Predicted Organ",
            organ["predicted_organ"],
            f"Confidence: {organ['confidence']:.1%}",
        )
        render_metric_card(
            "Calibration T",
            f"{organ['temperature']:.2f}",
            "Temperature scaling",
        )
        st.plotly_chart(build_organ_bar(organ["top3"]), use_container_width=True)
        match = "Match" if organ["predicted_organ"] == ground_truth else "Mismatch"
        cls = "finding-success" if organ["predicted_organ"] == ground_truth else "finding-error"
        st.markdown(
            f'<div class="finding {cls}">{match} Ground Truth: <b>{ground_truth}</b></div>',
            unsafe_allow_html=True,
        )

    mask = results["cells"].get("mask")
    detected_cells = results["cells"].get("detected", [])
    if mask is not None and mask.any() and detected_cells:
        st.markdown("### Segmentation Mask (Class-Specific)")
        overlay = image.copy().astype(np.float64)
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        elif overlay.ndim == 3 and overlay.shape[2] == 1:
            overlay = np.repeat(overlay, 3, axis=2)
        elif overlay.ndim == 3 and overlay.shape[2] > 3:
            overlay = overlay[:, :, :3]

        mask_ys, mask_xs = np.where(mask > 0)
        if len(mask_ys) > 0 and len(detected_cells) > 0:
            centroids_arr = np.array([[c["centroid"][0], c["centroid"][1]] for c in detected_cells])
            class_colors: List[List[int]] = []
            for c in detected_cells:
                hex_color = CELL_COLORS.get(c["class_name"], "#8d99ae")
                class_colors.append(
                    [int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]
                )
            class_colors_arr = np.array(class_colors)

            pixel_coords = np.stack([mask_xs, mask_ys], axis=1).astype(np.float64)

            chunk_size = 50000
            for i in range(0, len(pixel_coords), chunk_size):
                chunk = pixel_coords[i : i + chunk_size]
                dists = np.linalg.norm(
                    chunk[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :], axis=2
                )
                nearest = np.argmin(dists, axis=1)
                colors = class_colors_arr[nearest]
                ys_chunk = mask_ys[i : i + chunk_size]
                xs_chunk = mask_xs[i : i + chunk_size]
                overlay[ys_chunk, xs_chunk] = overlay[ys_chunk, xs_chunk] * 0.5 + colors * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        st.image(overlay, use_container_width=True)

        legend_items = "  ".join(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:2px;margin-right:4px;"></span>{name}'
            for name, color in CELL_COLORS.items()
        )
        st.markdown(
            f'<div style="margin-top:8px;font-size:0.85rem;">{legend_items}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Cell Detection & Classification")
    cells = results["cells"]

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Cells", str(cells["total_cells"]), "DINOv2 segmentation")
    with col2:
        top_cell = (
            max(cells["class_counts"], key=lambda k: cells["class_counts"][k])
            if cells["class_counts"]
            else "N/A"
        )
        render_metric_card(
            "Dominant Type",
            top_cell,
            f"{cells['class_counts'].get(top_cell, 0)} cells",
        )
    with col3:
        validity = cells["validity"]
        status = "Valid" if validity["is_valid"] else "Exploratory"
        render_metric_card("Validity Gate", status, f"Min: {validity['min_total_cells']} cells")

    col_pie, col_scatter = st.columns([1, 1])
    with col_pie:
        st.markdown("#### Cell Type Distribution")
        st.plotly_chart(build_cell_pie(cells["class_counts"]), use_container_width=True)
    with col_scatter:
        st.markdown("#### Spatial Distribution")
        st.plotly_chart(build_spatial_scatter(cells["centroids"]), use_container_width=True)

    # Spatial Graph
    st.markdown("### Spatial Graph Analytics")
    gm = cells["graph_metrics"]

    avg_degree = (
        (2.0 * gm.get("total_edges", 0) / gm["total_nodes"])
        if gm.get("total_nodes", 0) > 0
        else gm.get("avg_degree", 0.0)
    )
    components = gm.get("connected_components", gm.get("num_components", 0))

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card("Graph Nodes", str(gm.get("total_nodes", 0)), "Detected cells")
    with col_b:
        render_metric_card("Graph Edges", str(gm.get("total_edges", 0)), "Proximity links")
    with col_c:
        render_metric_card("Avg Degree", f"{avg_degree:.2f}", "Connectivity")
    with col_d:
        render_metric_card("Components", str(components), "Clusters")

    # Interaction matrix
    interactions = gm.get("interaction_matrix", {})
    if interactions:
        st.markdown("#### Cell-Cell Interactions")
        for pair, count in sorted(interactions.items(), key=lambda x: -x[1])[:8]:
            st.markdown(
                f'<div class="finding finding-info">{pair}: <b>{count}</b> interactions</div>',
                unsafe_allow_html=True,
            )

    # Immune infiltration
    immune = gm.get("immune_infiltration", {})
    if immune and immune.get("inflammatory_count", 0) > 0:
        st.markdown("#### Immune Infiltration")
        col_e, col_f = st.columns(2)
        with col_e:
            mean_px = immune.get("mean_immune_to_tumor_dist_px", 0)
            render_metric_card(
                "Mean Immune->Tumor",
                f"{mean_px:.1f} px ({mean_px * 0.25:.1f} um)",
                "Inflammatory -> Neoplastic",
            )
        with col_f:
            min_px = immune.get("closest_immune_to_tumor_dist_px", 0)
            render_metric_card(
                "Closest Immune->Tumor",
                f"{min_px:.1f} px ({min_px * 0.25:.1f} um)",
                "Minimum distance",
            )

    avg_dist_px = gm.get("avg_neighbor_distance_px", 0)
    if avg_dist_px > 0:
        st.markdown(
            f'<div style="margin-top:8px;font-size:0.8rem;color:#5f6368;">'
            f"Avg neighbor distance: <b>{avg_dist_px:.1f} px</b> ~ "
            f"<b>{avg_dist_px * 0.25:.1f} um</b> (at 40x, 0.25 um/px)</div>",
            unsafe_allow_html=True,
        )

    # Validity warnings
    if not cells["validity"]["is_valid"]:
        for w in cells["validity"]["warnings"][:3]:
            st.markdown(f'<div class="finding finding-warn">{w}</div>', unsafe_allow_html=True)

    # Timing
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f'<div style="text-align:center; margin-top:16px; color:#5f6368; font-size:0.72rem;">'
        f'Organ: {results["organ"]["time"]:.2f}s | '
        f'Cells+Spatial: {results["cells"]["time"]:.2f}s | Device: {device_str}</div>',
        unsafe_allow_html=True,
    )

    # CrewAI Report
    if not skip_ai_report:
        st.markdown("### CrewAI Multi-Agent Report")
        st.info("Launching full CrewAI multi-agent pipeline... This may take 30-90 seconds.")

        # Progress bar & live agent status
        task_names = [
            "Image Parsing",
            "Spatial Analysis",
            "Literature Research",
            "Report Synthesis",
        ]
        progress_bar = st.progress(0, text="Starting...")
        task_status = st.empty()

        def on_task_complete(output: Any) -> None:
            """Callback invoked after each task completes."""
            task_idx = 0
            if "spatial" in str(output).lower() or "graph" in str(output).lower():
                task_idx = 1
            elif (
                "literature" in str(output).lower()
                or "citation" in str(output).lower()
                or "pmid" in str(output).lower()
            ):
                task_idx = 2
            elif "report" in str(output).lower() or "section" in str(output).lower():
                task_idx = 3

            progress = (task_idx + 1) / 4
            progress_bar.progress(
                progress, text=f"{task_names[task_idx]} complete ({task_idx + 1}/4)"
            )
            task_status.markdown(
                f"**Current:** {task_names[min(task_idx + 1, 3)] if task_idx < 3 else 'Finalizing...'}"
            )

        try:
            report = run_crewai_pipeline(
                fold_dir, patch_idx, results, progress_callback=on_task_complete
            )
            progress_bar.progress(1.0, text="All tasks complete!")
            st.session_state["crewai_report"] = report
        except Exception as e:
            st.error(f"CrewAI pipeline failed: {str(e)}")
            st.code(str(e))
            return

    if "crewai_report" in st.session_state:
        st.markdown("#### Agent Pipeline")
        render_agent_pipeline(completed=True)
        st.markdown("#### Clinical Report")
        st.markdown(st.session_state["crewai_report"])

    # --- Chat with the Slide ---
    st.markdown("### Chat with the Slide")
    st.caption(
        "Ask free-form questions about this patch. The agent answers strictly from computed data."
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions
    if not st.session_state["chat_history"]:
        suggestions = [
            "Why is this classified as an Immune Desert?",
            "What's the dominant cell type and why?",
            "Is this sample statistically valid?",
            "What's the immune infiltration depth?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_input_val"] = s

    # Chat input
    user_input = st.chat_input("Ask about this slide... e.g. 'Why is this an immune desert?'")
    # Handle suggested question click
    if not user_input and "chat_input_val" in st.session_state:
        user_input = st.session_state.pop("chat_input_val")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    answer = run_chat_query(
                        user_input, results, st.session_state["chat_history"][:-1]
                    )
                    st.markdown(answer)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": f"Error: {e}"}
                    )
                    st.error(f"{e}")

    # Footer
    st.divider()
    st.markdown(
        f'<div style="text-align:center; color:#5f6368; font-size:0.72rem;">'
        f'BioCellAgents v2.0 | PanNuke + DINOv2 + CrewAI | {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
