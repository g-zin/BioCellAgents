"""
BioCellAgents - Computational Pathology Report
================================================
Multi-agent deep-learning pipeline for H&E histopathology analysis.
Produces a professional clinical-style report with spatial analytics.

Run: streamlit run app.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from datetime import datetime

import fpdf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw

from src.data.dataset import CELL_TYPES
from src.dashboard.charts import (
    build_confidence_histogram,
    build_size_histogram,
    build_interaction_heatmap,
    build_network_graph,
)
from src.dashboard.explorer import render_cell_explorer
from src.dashboard.styles import CELL_COLORS
from src.inference.calibration import TemperatureScaler
from src.inference.validity_gate import ValidityGate
from src.models.cellpose_segmenter import CellposeSegmenter
from src.models.dino_classifier import DINOv2CellClassifier
from src.models.inference_pipeline import TwoStageCellAnalyzer

st.set_page_config(
    page_title="BioCellAgents - Computational Pathology",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSIFIER_CKPT = PROJECT_ROOT / "checkpoints" / "cell_vits14" / "best_model.pt"
FOLD3_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "fold3" / "images"
SAMPLE_PATCH_IDS = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]
APP_VERSION = "1.0.0"


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* --- Base --- */
.stApp { background: #f1f5f9; }
.block-container { padding-top: 2rem; }

/* --- Clinical header --- */
.report-header {
    border-bottom: 4px solid #1e3a5f;
    padding: 24px 28px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
    border-radius: 10px;
    border: 1px solid #bfdbfe;
}
.report-title {
    font-size: 1.8rem; font-weight: 800; color: #0f172a;
    margin-bottom: 6px; letter-spacing: -0.03em;
}
.report-subtitle { font-size: 0.95rem; color: #1e293b; font-weight: 600; }
.report-meta {
    display: flex; gap: 28px; margin-top: 14px;
    font-size: 0.85rem; color: #475569; font-weight: 500;
}
.report-meta span { white-space: nowrap; }

/* --- Section headings --- */
h2 {
    font-size: 1.25rem; font-weight: 800; color: #0f172a;
    border-bottom: 3px solid #3b82f6; padding-bottom: 10px;
    margin-top: 36px;
}
h3 { font-size: 1.1rem; font-weight: 700; color: #1e293b; }
h4 { font-size: 0.95rem; font-weight: 700; color: #334155; }

/* --- Finding blocks --- */
.finding-block {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-left: 5px solid #2563eb; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; font-size: 0.92rem;
    color: #1e293b; line-height: 1.7; font-weight: 500;
}
.finding-block.warning { border-left-color: #d97706; background: #fffbeb; border-color: #fbbf24; }
.finding-block.critical { border-left-color: #dc2626; background: #fef2f2; border-color: #f87171; }
.finding-block.success { border-left-color: #059669; background: #ecfdf5; border-color: #34d399; }
.finding-block.error { border-left-color: #dc2626; background: #fef2f2; border-color: #f87171; }

/* --- Metrics --- */
div[data-testid="stMetric"] {
    background: #ffffff; border: 2px solid #cbd5e1;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label { color: #475569 !important; font-size: 0.82rem !important; font-weight: 600 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800 !important; }

/* --- Tables --- */
div[data-testid="stDataFrame"] {
    border: 2px solid #cbd5e1; border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* --- Buttons --- */
.stButton > button {
    border-radius: 8px !important; font-weight: 700 !important;
    border: 2px solid #94a3b8 !important; background: #ffffff !important;
    color: #1e293b !important; padding: 10px 18px !important;
}
.stButton > button:hover { background: #f1f5f9 !important; border-color: #64748b !important; }
.stButton > button[kind="primary"] {
    background: #1d4ed8 !important; color: #fff !important;
    border: none !important; font-size: 0.95rem !important;
}
.stButton > button[kind="primary"]:hover { background: #1e40af !important; }

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e2e8f0; border-radius: 10px; padding: 5px;
    border: 2px solid #cbd5e1;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; font-weight: 700; font-size: 0.9rem;
    color: #334155;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    border: 2px solid #94a3b8;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    color: #1e3a5f !important;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 3px solid #cbd5e1;
}
/* Force high-contrast text in the sidebar (Streamlit theme otherwise
   overrides the base h2/p/label rules with a light grey secondary color). */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #0f172a !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #1e293b !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #1e293b !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stSliderValue"] {
    color: #0f172a !important; font-weight: 700 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    color: #1e293b !important;
}
section[data-testid="stSidebar"] .stButton > button {
    color: #1e293b !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] summary {
    color: #1e3a5f !important;
}

/* --- Alerts --- */
.stAlert { border-radius: 8px; border-width: 2px; }

/* --- Dividers --- */
hr { border-color: #cbd5e1; }

/* --- Report section --- */
.report-section {
    background: #ffffff; border: 2px solid #cbd5e1;
    border-radius: 10px; padding: 22px 26px; margin: 14px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.report-section h4 { margin-bottom: 12px; color: #1e3a5f; font-size: 1.05rem; font-weight: 700; }
.report-section p { color: #1e293b; line-height: 1.7; font-size: 0.95rem; }

/* --- Image gallery --- */
.patch-thumb {
    border: 3px solid #cbd5e1; border-radius: 8px;
    cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s;
}
.patch-thumb:hover { border-color: #2563eb; box-shadow: 0 3px 12px rgba(37,99,235,0.25); }
.patch-label {
    font-size: 0.8rem; color: #334155; text-align: center;
    margin-top: 5px; font-weight: 700;
}

/* --- Code / inline --- */
code { background: #e2e8f0; color: #1e3a5f; padding: 3px 7px; border-radius: 5px; font-size: 0.88rem; font-weight: 600; }

/* --- General text contrast --- */
p, span, label { color: #1e293b; }
.stMarkdown p { color: #1e293b; font-weight: 500; }
.stCaption, [data-testid="stCaptionContainer"] { color: #475569 !important; font-weight: 600; }

/* --- Dataframe / table headers --- */
div[data-testid="stDataFrame"] [data-testid="stDataFrameHeader"] {
    background: #1e3a5f !important; color: #ffffff !important; font-weight: 700;
}
div[data-testid="stDataFrame"] th {
    background: #1e3a5f !important; color: #ffffff !important; font-weight: 700;
}
div[data-testid="stDataFrame"] td { color: #1e293b !important; font-weight: 500; }

/* --- Expander --- */
details[data-testid="stExpander"] summary {
    font-weight: 700 !important; color: #1e3a5f !important;
    border: 2px solid #cbd5e1; border-radius: 8px; padding: 10px 16px;
    background: #f8fafc;
}
details[data-testid="stExpander"] summary:hover { background: #e2e8f0; }

/* --- Form elements --- */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] {
    border: 2px solid #94a3b8 !important; border-radius: 8px !important;
    background: #ffffff !important;
}
div[data-baseweb="select"] span { color: #1e293b !important; font-weight: 600; }
input[type="range"] { accent-color: #1d4ed8; }
div[data-testid="stSlider"] { border: 2px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; background: #ffffff; }

/* --- Progress bar --- */
div[data-testid="stProgress"] > div > div > div {
    background: #1d4ed8 !important;
}
div[data-testid="stProgress"] span { color: #1e293b !important; font-weight: 700; }

/* --- File uploader --- */
div[data-testid="stFileUploader"] {
    border: 2px dashed #94a3b8 !important; border-radius: 10px !important;
    background: #f8fafc !important;
}
div[data-testid="stFileUploader"] span { color: #334155 !important; font-weight: 600; }

/* --- Divider --- */
hr { border: 2px solid #cbd5e1 !important; }

/* --- Strong / bold --- */
strong, b { color: #0f172a; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_analyzer(device: str = "cpu") -> TwoStageCellAnalyzer:
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    segmenter = CellposeSegmenter(device=str(torch_device))
    classifier = DINOv2CellClassifier(
        model_size="vits14", num_classes=len(CELL_TYPES), freeze_backbone=True,
    )
    if CLASSIFIER_CKPT.exists():
        state_dict = torch.load(CLASSIFIER_CKPT, map_location=torch_device, weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        classifier.load_state_dict(state_dict)
    classifier.to(torch_device).eval()
    calibration = TemperatureScaler(default_temperature=2.0)
    return TwoStageCellAnalyzer(
        segmenter, classifier, device=str(torch_device), calibration_scaler=calibration
    )


class AgentFinding:
    def __init__(self, text: str, severity: str = "info"):
        self.text = text
        self.severity = severity


class SegmentationAgent:
    name = "Cell Segmentation"

    def analyze(self, mask, cells, image_size):
        findings = []
        total = len(cells)
        fg_ratio = np.mean(mask > 0) if mask.size > 0 else 0
        findings.append(AgentFinding(
            f"Detected {total} cellular structures in a {image_size[0]}\u00d7{image_size[1]} px field. "
            f"Foreground occupancy: {100*fg_ratio:.1f}%.",
            "success" if total > 0 else "warning",
        ))
        if total > 0:
            centroids = np.array([c["centroid"] for c in cells])
            if len(centroids) > 2:
                mean_dist = np.mean(np.linalg.norm(
                    centroids[:, None] - centroids[None, :], axis=-1))
                if mean_dist > 20:
                    findings.append(AgentFinding(
                        f"Mean inter-cellular distance: {mean_dist:.1f} px. "
                        "Cellular distribution is uniform across the field.",
                        "info"))
                else:
                    findings.append(AgentFinding(
                        f"Mean inter-cellular distance: {mean_dist:.1f} px. "
                        "High cellular density observed; cells are closely packed.",
                        "info"))
        else:
            findings.append(AgentFinding(
                "No cellular structures detected. Review image quality and staining.",
                "critical"))
        return findings


class ClassificationAgent:
    name = "Cell Classification"

    def analyze(self, cells):
        findings = []
        if not cells:
            findings.append(AgentFinding("No cells available for classification.", "critical"))
            return findings
        counts = {ct: 0 for ct in CELL_TYPES}
        for c in cells:
            if c["class_name"] in counts:
                counts[c["class_name"]] += 1
        total = len(cells)
        dominant = max(counts, key=counts.get)
        findings.append(AgentFinding(
            f"Predominant cell type: {dominant} "
            f"({counts[dominant]}/{total} = {100*counts[dominant]/total:.0f}%).",
            "info"))
        confs = [c["confidence"] for c in cells]
        mean_conf = np.mean(confs)
        low_conf = sum(1 for c in confs if c < 0.5)
        findings.append(AgentFinding(
            f"Mean classification confidence: {mean_conf:.2f}. "
            f"{low_conf}/{total} cells below 0.50 threshold.",
            "success" if mean_conf > 0.7 else "warning"))
        dist_str = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0)
        findings.append(AgentFinding(f"Cellular composition \u2014 {dist_str}.", "info"))
        return findings


class MorphologyAgent:
    name = "Morphometric Analysis"

    def analyze(self, cells, image_size):
        findings = []
        if not cells:
            findings.append(AgentFinding("Insufficient data for morphometric analysis.", "warning"))
            return findings
        centroids = np.array([c["centroid"] for c in cells])
        h, w = image_size[1], image_size[0]
        mid_x, mid_y = w / 2, h / 2
        q1 = sum(1 for c in centroids if c[0] < mid_x and c[1] < mid_y)
        q2 = sum(1 for c in centroids if c[0] >= mid_x and c[1] < mid_y)
        q3 = sum(1 for c in centroids if c[0] < mid_x and c[1] >= mid_y)
        q4 = sum(1 for c in centroids if c[0] >= mid_x and c[1] >= mid_y)
        if max(q1, q2, q3, q4) < 0.6 * len(cells):
            findings.append(AgentFinding(
                f"Quadrant distribution \u2014 TL: {q1}, TR: {q2}, BL: {q3}, BR: {q4}. "
                "Cellular distribution is even across the field.",
                "info"))
        else:
            findings.append(AgentFinding(
                f"Quadrant distribution \u2014 TL: {q1}, TR: {q2}, BL: {q3}, BR: {q4}. "
                "Cells are concentrated in specific regions.",
                "info"))
        density = len(cells) / (w * h) * 10000
        if density > 5:
            findings.append(AgentFinding(
                f"Cellular density: {density:.1f} cells/10k px\u00b2. High density observed.",
                "info"))
        else:
            findings.append(AgentFinding(
                f"Cellular density: {density:.1f} cells/10k px\u00b2. "
                "Low density \u2014 stromal or necrotic tissue possible.",
                "info"))
        return findings


class ValidityAgent:
    name = "Quality Control"

    def analyze(self, cells):
        findings = []
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        counts = {ct: sum(1 for c in cells if c["class_name"] == ct) for ct in CELL_TYPES}
        assessment = gate.full_assessment(len(cells), counts)
        if assessment["is_valid"]:
            findings.append(AgentFinding(
                f"QC Status: PASS ({len(cells)} cells >= threshold {gate.min_total_cells}). "
                "Results are statistically reliable.",
                "success"))
        else:
            findings.append(AgentFinding(
                f"QC Status: BELOW MINIMUM THRESHOLD ({len(cells)} < {gate.min_total_cells}). "
                "Results are exploratory and should not be used for diagnostic purposes.",
                "critical"))
        for w in assessment.get("warnings", []):
            findings.append(AgentFinding(w, "warning"))
        for r in assessment.get("recommendations", []):
            findings.append(AgentFinding(r, "info"))
        return findings



class SpatialAnalyticsAgent:
    """Computes pairwise distances, proximity graph, and graph topology metrics."""
    name = "Spatial Analysis"
    EDGE_THRESHOLD = 50

    def analyze(self, cells):
        if len(cells) < 2:
            return {
                "pairwise_matrix": None, "graph": None,
                "metrics": {"density": 0, "avg_degree": 0, "clustering_coeff": 0,
                            "n_nodes": len(cells), "n_edges": 0},
                "tme": {"inflammatory_neoplastic_mean": None,
                        "inflammatory_neoplastic_min": None,
                        "inflammatory_neoplastic_max": None,
                        "inflammatory_neoplastic_count": 0},
                "type_pairs": [],
            }
        centroids = np.array([c["centroid"] for c in cells], dtype=np.float64)
        labels = [c["class_name"] for c in cells]
        n = len(centroids)
        diff = centroids[:, None, :] - centroids[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)
        adjacency = (dist_matrix < self.EDGE_THRESHOLD).astype(np.int32)
        np.fill_diagonal(adjacency, 0)
        n_edges = int(adjacency.sum()) // 2
        max_edges = n * (n - 1) // 2
        density = n_edges / max_edges if max_edges > 0 else 0.0
        degrees = adjacency.sum(axis=1)
        avg_degree = float(degrees.mean())
        cc_values = []
        for i in range(n):
            neighbors = np.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                cc_values.append(0.0)
                continue
            sub = adjacency[np.ix_(neighbors, neighbors)]
            links = sub.sum() / 2
            cc_values.append(2.0 * links / (k * (k - 1)))
        clustering_coeff = float(np.mean(cc_values)) if cc_values else 0.0
        inf_idx = [i for i, l in enumerate(labels) if l == "Inflammatory"]
        neo_idx = [i for i, l in enumerate(labels) if l == "Neoplastic"]
        tme = {"inflammatory_neoplastic_mean": None,
               "inflammatory_neoplastic_min": None,
               "inflammatory_neoplastic_max": None,
               "inflammatory_neoplastic_count": 0}
        if inf_idx and neo_idx:
            inf_neo_dists = dist_matrix[np.ix_(inf_idx, neo_idx)]
            tme = {"inflammatory_neoplastic_mean": float(inf_neo_dists.mean()),
                   "inflammatory_neoplastic_min": float(inf_neo_dists.min()),
                   "inflammatory_neoplastic_max": float(inf_neo_dists.max()),
                   "inflammatory_neoplastic_count": int(inf_neo_dists.size)}
        type_pairs = []
        for i, t1 in enumerate(CELL_TYPES):
            for t2 in CELL_TYPES[i + 1:]:
                idx1 = [j for j, l in enumerate(labels) if l == t1]
                idx2 = [j for j, l in enumerate(labels) if l == t2]
                if idx1 and idx2:
                    d = dist_matrix[np.ix_(idx1, idx2)]
                    type_pairs.append({"pair": f"{t1} <-> {t2}", "mean_dist": float(d.mean()),
                                       "min_dist": float(d.min()), "n_pairs": int(d.size)})
        return {
            "pairwise_matrix": dist_matrix, "graph": adjacency,
            "metrics": {"density": density, "avg_degree": avg_degree,
                        "clustering_coeff": clustering_coeff,
                        "n_nodes": n, "n_edges": n_edges},
            "tme": tme, "type_pairs": type_pairs,
        }



class DiagnosticAgent:
    name = "Report Synthesis"

    def analyze(self, cells, spatial_results=None, validity_findings=None):
        report = {
            "executive_summary": "", "cellular_breakdown": [],
            "tme_metrics": {}, "pairwise_matrix": [],
            "graph_topology": {}, "validity": "", "limitations": [],
        }
        if not cells:
            report["executive_summary"] = (
                "No cellular structures detected in this field. "
                "The patch may contain non-tissue regions or artifacts. "
                "Review image quality and consider re-imaging.")
            report["limitations"] = ["Insufficient cell detections for meaningful analysis."]
            return report
        counts = {ct: sum(1 for c in cells if c["class_name"] == ct) for ct in CELL_TYPES}
        total = len(cells)
        dominant = max(counts, key=counts.get)
        mean_conf = np.mean([c["confidence"] for c in cells])
        summary = (f"Analysis of {total} cells identified {sum(1 for v in counts.values() if v > 0)} "
                   f"cell types. {dominant}-predominant "
                   f"({100*counts[dominant]/total:.0f}%). ")
        if counts.get("Neoplastic", 0) > total * 0.3:
            summary += (
                "Elevated neoplastic proportion warrants further investigation. "
                "Correlation with clinical history and immunohistochemical panel is recommended.")
        elif counts.get("Inflammatory", 0) > total * 0.4:
            summary += (
                "Inflammatory predominance is consistent with active immune response. "
                "Differential includes infection, autoimmune process, or tumor-associated inflammation.")
        else:
            summary += (
                "Cellular composition is within the expected range for this tissue type. "
                "No immediate pathological concern identified from this field.")
        report["executive_summary"] = summary
        for ct in CELL_TYPES:
            if counts[ct] > 0:
                report["cellular_breakdown"].append({
                    "type": ct, "count": counts[ct],
                    "pct": f"{100*counts[ct]/total:.1f}%",
                })
        if spatial_results and spatial_results.get("tme"):
            tme = spatial_results["tme"]
            report["tme_metrics"] = {
                "Inflammatory <-> Neoplastic (mean px)": tme.get("inflammatory_neoplastic_mean"),
                "Inflammatory <-> Neoplastic (min px)": tme.get("inflammatory_neoplastic_min"),
                "Inflammatory <-> Neoplastic (max px)": tme.get("inflammatory_neoplastic_max"),
                "Immune cell proportion": f"{100*counts.get('Inflammatory',0)/total:.1f}%",
                "Neoplastic proportion": f"{100*counts.get('Neoplastic',0)/total:.1f}%",
            }
        if spatial_results and spatial_results.get("type_pairs"):
            for tp in spatial_results["type_pairs"]:
                report["pairwise_matrix"].append(tp)
        if spatial_results and spatial_results.get("metrics"):
            m = spatial_results["metrics"]
            report["graph_topology"] = {
                "Nodes (cells)": m["n_nodes"], "Edges (<50px)": m["n_edges"],
                "Graph Density": f"{m['density']:.4f}",
                "Avg Degree": f"{m['avg_degree']:.2f}",
                "Clustering Coefficient": f"{m['clustering_coeff']:.4f}",
            }
        if validity_findings:
            report["validity"] = " | ".join(f.text for f in validity_findings[:2])
        else:
            report["validity"] = f"{total} cells analyzed."
        report["limitations"] = [
            "Single-patch analysis; multi-region correlation is recommended.",
            "2D projection of 3D tissue architecture.",
            "Classification confidence varies; manual review is advised for low-confidence cells.",
            "Spatial metrics are pixel-based; calibration to \u00b5m requires magnification metadata.",
            "No immunohistochemical or molecular data incorporated.",
        ]
        if mean_conf < 0.6:
            report["limitations"].insert(0,
                f"Mean confidence {mean_conf:.2f} is moderate \u2014 interpret results with caution.")
        return report


def run_multi_agent_analysis(mask, cells, image_size):
    seg_agent = SegmentationAgent()
    cls_agent = ClassificationAgent()
    morph_agent = MorphologyAgent()
    val_agent = ValidityAgent()
    spatial_agent = SpatialAnalyticsAgent()
    diag_agent = DiagnosticAgent()
    seg_findings = seg_agent.analyze(mask, cells, image_size)
    cls_findings = cls_agent.analyze(cells)
    morph_findings = morph_agent.analyze(cells, image_size)
    val_findings = val_agent.analyze(cells)
    spatial_results = spatial_agent.analyze(cells)
    diag_report = diag_agent.analyze(cells, spatial_results, val_findings)
    agent_results = {
        "Cell Segmentation": seg_findings,
        "Cell Classification": cls_findings,
        "Morphometric Analysis": morph_findings,
        "Spatial Analysis": [],
        "Quality Control": val_findings,
        "Report Synthesis": [],
    }
    return agent_results, spatial_results, diag_report



def get_sample_images():
    samples = []
    for pid in SAMPLE_PATCH_IDS:
        path = FOLD3_IMAGES_DIR / f"patch_{pid:04d}.png"
        if path.exists():
            img = Image.open(path).convert("RGB")
            samples.append({"id": pid, "path": path, "image": img})
    return samples


def load_ground_truth(patch_id):
    """Load ground truth cell labels from crop filenames for a given patch.
    Crop naming: cell_{patch:04d}_c{class}_inst{instance:03d}.png
    Returns list of dicts: [{'class_idx': int, 'class_name': str, 'instance': int}, ...]
    """
    crops_dir = FOLD3_IMAGES_DIR.parent / "crops"
    if not crops_dir.exists():
        return []
    gt_cells = []
    prefix = f"cell_{patch_id:04d}_"
    for f in sorted(crops_dir.iterdir()):
        if f.name.startswith(prefix) and f.suffix == ".png":
            # Parse: cell_0000_c2_inst015.png
            parts = f.stem.split("_")
            # parts = ['cell', '0000', 'c2', 'inst015']
            try:
                class_idx = int(parts[2][1:])  # 'c2' -> 2
                instance = int(parts[3][4:])  # 'inst015' -> 15
                gt_cells.append({
                    "class_idx": class_idx,
                    "class_name": CELL_TYPES[class_idx] if class_idx < len(CELL_TYPES) else "Unknown",
                    "instance": instance,
                })
            except (IndexError, ValueError):
                continue
    return gt_cells


def compare_predictions(cells, gt_cells):
    """Compare predicted cell classifications against ground truth.
    Returns a list of comparison dicts and summary stats.
    """
    if not gt_cells:
        return None
    # Build GT class counts
    gt_counts = {ct: 0 for ct in CELL_TYPES}
    for gt in gt_cells:
        if gt["class_name"] in gt_counts:
            gt_counts[gt["class_name"]] += 1

    # Build predicted class counts
    pred_counts = {ct: 0 for ct in CELL_TYPES}
    for cell in cells:
        if cell["class_name"] in pred_counts:
            pred_counts[cell["class_name"]] += 1

    # Per-class comparison
    comparison = []
    for ct in CELL_TYPES:
        gt_n = gt_counts.get(ct, 0)
        pred_n = pred_counts.get(ct, 0)
        comparison.append({
            "class": ct,
            "ground_truth": gt_n,
            "predicted": pred_n,
            "diff": pred_n - gt_n,
            "match": "Yes" if gt_n == pred_n else "No",
        })

    # Overall accuracy (proportion match)
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    # Simple proportion-based accuracy
    if total_gt > 0 and total_pred > 0:
        # Use min overlap as a proxy for agreement
        overlap = sum(min(gt_counts.get(ct, 0), pred_counts.get(ct, 0)) for ct in CELL_TYPES)
        agreement = overlap / max(total_gt, total_pred)
    else:
        agreement = 0.0

    return {
        "comparison": comparison,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "agreement": round(agreement, 3),
    }


def annotate_image(image, cells):
    """Create a high-resolution annotated image with colored ellipses, center dots, and connection lines."""
    annotated = image.copy().convert("RGB")
    scale = 3
    big = annotated.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(big)

    # Connection lines between nearby centers
    centers = [(c["centroid"][0] * scale, c["centroid"][1] * scale) for c in cells]
    max_dist = 80 * scale
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 < max_dist:
                draw.line([centers[i], centers[j]], fill=(148, 163, 184, 100), width=1)

    # Colored ellipses + center dots + labels
    for cell in cells:
        x1, y1, x2, y2 = [v * scale for v in cell["bbox"]]
        color = CELL_COLORS.get(cell["class_name"], "#FFFFFF")
        draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
        # Center dot
        cx, cy = int(cell["centroid"][0] * scale), int(cell["centroid"][1] * scale)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
        # Label
        label = f"{cell['class_name'][:4]} {cell['confidence']:.0%}"
        draw.text((x1 + 6, max(4, y1 - 20)), label, fill=color)
    return big


def create_overlay(image, mask, cells, alpha=0.55):
    """Create a professional overlay: transparent colored segmentation + center markers + connection lines."""
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw transparent colored regions per cell (using bbox as approximation of mask region)
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        color_hex = CELL_COLORS.get(cell["class_name"], "#FFFFFF")
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        # Semi-transparent fill
        draw.ellipse([x1, y1, x2, y2], fill=(r, g, b, int(255 * alpha * 0.6)))
        # Subtle border
        draw.ellipse([x1, y1, x2, y2], outline=(r, g, b, 180), width=1)

    # Draw connection lines between nearby cell centers (spatial graph)
    centers = [(c["centroid"][0], c["centroid"][1]) for c in cells]
    max_dist = 80  # px threshold for drawing connections
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < max_dist:
                draw.line([centers[i], centers[j]], fill=(100, 116, 139, 80), width=1)

    # Draw center dots
    for cell in cells:
        cx, cy = int(cell["centroid"][0]), int(cell["centroid"][1])
        color_hex = CELL_COLORS.get(cell["class_name"], "#FFFFFF")
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(r, g, b, 220), outline=(255, 255, 255, 200), width=1)

    blended = Image.alpha_composite(base, overlay)
    return blended.convert("RGB")


def mask_to_display(mask):
    display = np.zeros((*mask.shape, 3), dtype=np.uint8)
    display[mask > 0] = [43, 108, 176]
    display[mask == 0] = [247, 250, 252]
    return display


def build_charts(cells):
    counts = {ct: 0 for ct in CELL_TYPES}
    for cell in cells:
        if cell["class_name"] in counts:
            counts[cell["class_name"]] += 1
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [CELL_COLORS.get(l, "#888") for l in labels]
    pie = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent", textfont=dict(size=13, color="#1e293b")))
    pie.update_layout(
        title="Cell Type Distribution", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b", size=13), height=350,
        margin=dict(t=50, b=20, l=20, r=20))
    bar = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
        text=values, textposition="outside", textfont=dict(color="#1e293b")))
    bar.update_layout(
        title="Cell Counts", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b", size=13), height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#cbd5e1"))
    return pie, bar


def build_spatial_scatter(cells, spatial_results):
    if not cells or len(cells) < 2:
        return None
    fig = go.Figure()
    for ct in CELL_TYPES:
        subset = [c for c in cells if c["class_name"] == ct]
        if not subset:
            continue
        xs = [c["centroid"][0] for c in subset]
        ys = [c["centroid"][1] for c in subset]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=ct,
            marker=dict(size=8, color=CELL_COLORS.get(ct, "#888")),
            text=[f"{c['cell_id']} ({c['confidence']:.0%})" for c in subset],
        ))
    if spatial_results and spatial_results.get("graph") is not None:
        adj = spatial_results["graph"]
        centroids = np.array([c["centroid"] for c in cells])
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if adj[i, j] > 0:
                    fig.add_trace(go.Scatter(
                        x=[centroids[i][0], centroids[j][0]],
                        y=[centroids[i][1], centroids[j][1]],
                        mode="lines", name="edge", showlegend=False,
                        line=dict(width=1, color="rgba(100,140,255,0.3)"),
                    ))
    fig.update_layout(
        title="Spatial Distribution & Proximity Graph",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13), height=500,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(title="X (px)", gridcolor="#cbd5e1"),
        yaxis=dict(title="Y (px)", gridcolor="#cbd5e1", scaleanchor="x", scaleratio=1),
    )
    return fig


def build_dataframe(cells):
    rows = []
    for cell in cells:
        rows.append({
            "ID": cell["cell_id"], "Class": cell["class_name"],
            "Conf": f"{cell['confidence']:.3f}",
            "BBox": str(cell["bbox"]), "Centroid": str(cell["centroid"]),
        })
    return pd.DataFrame(rows)


def export_json(cells, image_size, source="", diag_report=None, spatial_results=None):
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    metadata = {
        "app_version": APP_VERSION,
        "pipeline": "Cellpose Segmentation + DINOv2 ViT-S/14 Classification",
        "dataset": "PanNuke (H&E)",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    payload = {
        "metadata": metadata,
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "image_size": list(image_size),
        "total_cells": len(cells),
        "cell_type_counts": {ct: sum(1 for c in cells if c["class_name"] == ct) for ct in CELL_TYPES},
        "cells": cells,
    }
    if diag_report:
        payload["report"] = {
            "executive_summary": diag_report.get("executive_summary", ""),
            "cellular_breakdown": diag_report.get("cellular_breakdown", []),
            "tme_metrics": diag_report.get("tme_metrics", {}),
            "pairwise_matrix": diag_report.get("pairwise_matrix", []),
            "graph_topology": diag_report.get("graph_topology", {}),
            "validity": diag_report.get("validity", ""),
            "limitations": diag_report.get("limitations", []),
        }
    if spatial_results and spatial_results.get("metrics"):
        payload["spatial_metrics"] = spatial_results["metrics"]
    return json.dumps(payload, indent=2, cls=NpEnc)


def export_csv(cells):
    return build_dataframe(cells).to_csv(index=False)


def generate_pdf_report(diag_report, spatial_results, cells, image_size, source):
    """Generate a professional clinical PDF report using fpdf2."""
    pdf = fpdf.FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Header ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(26, 35, 50)
    pdf.cell(0, 12, "Computational Pathology Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 6, "BioCellAgents - Multi-Module Analysis Pipeline", ln=True, align="C")
    pdf.ln(4)

    # --- Metadata block ---
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(113, 128, 150)
    meta_lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Image: {source} ({image_size[0]}x{image_size[1]} px)",
        f"Total Cells: {len(cells)}",
        f"Pipeline: Cellpose + DINOv2 ViT-S/14",
        f"Version: {APP_VERSION}",
    ]
    for line in meta_lines:
        pdf.cell(0, 5, line, ln=True)
    pdf.ln(4)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # --- Section helpers ---
    def _pdf_safe(text):
        """Replace characters not supported by Helvetica (Latin-1) with ASCII equivalents."""
        replacements = {
            '\u2194': '<->', '\u2265': '>=', '\u2264': '<=',
            '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"', '\u2026': '...',
            '\u00b5': 'u', '\u00b2': '2', '\u00b3': '3',
            '\u00d7': 'x', '\u2260': '!=', '\u2248': '~',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Fallback: strip any remaining non-Latin-1 chars
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def section_title(title):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(26, 35, 50)
        pdf.cell(0, 8, _pdf_safe(title), ln=True)
        pdf.ln(1)

    def body_text(text, indent=0):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(45, 55, 72)
        pdf.set_x(15 + indent)
        pdf.multi_cell(0, 5, _pdf_safe(text))
        pdf.ln(2)

    def table(headers, rows):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(247, 250, 252)
        pdf.set_text_color(26, 35, 50)
        col_w = (180 - 10) / len(headers)
        for h in headers:
            pdf.cell(col_w, 6, _pdf_safe(h), border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(45, 55, 72)
        for row in rows:
            for i, val in enumerate(row):
                pdf.cell(col_w, 5, _pdf_safe(str(val)), border=1, align="C")
            pdf.ln()
        pdf.ln(3)

    # --- 1. Executive Summary ---
    section_title("1. Executive Summary")
    body_text(diag_report.get("executive_summary", "N/A"))

    # --- 2. Cellular Breakdown ---
    section_title("2. Cellular Breakdown")
    breakdown = diag_report.get("cellular_breakdown", [])
    if breakdown:
        table(["Cell Type", "Count", "Proportion"],
              [[r["type"], r["count"], r["pct"]] for r in breakdown])
    else:
        body_text("No cellular data available.")

    # --- 3. TME Metrics ---
    section_title("3. Tumor Microenvironment (TME) Metrics")
    tme = diag_report.get("tme_metrics", {})
    if tme:
        table(["Metric", "Value"], [[k, v] for k, v in tme.items()])
    else:
        body_text("Insufficient data for TME analysis.")

    # --- 4. Pairwise Interaction Matrix ---
    section_title("4. Pairwise Interaction Matrix")
    pairs = diag_report.get("pairwise_matrix", [])
    if pairs:
        table(["Cell Pair", "Mean Dist (px)", "Min Dist (px)", "N Pairs"],
              [[p["pair"], f"{p['mean_dist']:.1f}", f"{p['min_dist']:.1f}", p["n_pairs"]] for p in pairs])
    else:
        body_text("Insufficient data for pairwise analysis.")

    # --- 5. Spatial Graph Topology ---
    section_title("5. Spatial Graph Topology")
    topo = diag_report.get("graph_topology", {})
    if topo:
        table(["Metric", "Value"], [[k, v] for k, v in topo.items()])
    else:
        body_text("Insufficient data for graph analysis.")

    # --- 6. Validity Assessment ---
    section_title("6. Validity Assessment")
    body_text(diag_report.get("validity", "N/A"))

    # --- 7. Limitations ---
    section_title("7. Limitations & Recommendations")
    for i, lim in enumerate(diag_report.get("limitations", []), 1):
        body_text(f"{i}. {lim}")

    # --- Footer ---
    pdf.ln(8)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 5, f"Report generated by BioCellAgents v{APP_VERSION} | "
                   f"Pipeline: Cellpose + DINOv2 ViT-S/14 | "
                   f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

    return bytes(pdf.output())



def generate_markdown_report(diag_report, spatial_results, cells, image_size, source):
    lines = []
    lines.append("# Computational Pathology Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Image:** {source} ({image_size[0]}x{image_size[1]} px)")
    lines.append(f"**Total Cells:** {len(cells)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(diag_report.get("executive_summary", "N/A"))
    lines.append("")
    lines.append("## 2. Cellular Breakdown")
    lines.append("")
    lines.append("| Cell Type | Count | Proportion |")
    lines.append("|-----------|-------|------------|")
    for row in diag_report.get("cellular_breakdown", []):
        lines.append(f"| {row['type']} | {row['count']} | {row['pct']} |")
    lines.append("")
    lines.append("## 3. Tumor Microenvironment (TME) Metrics")
    lines.append("")
    tme = diag_report.get("tme_metrics", {})
    if tme:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in tme.items():
            val = f"{v:.1f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {val} |")
    else:
        lines.append("_Insufficient data for TME analysis._")
    lines.append("")
    lines.append("## 4. Pairwise Interaction Matrix")
    lines.append("")
    pairs = diag_report.get("pairwise_matrix", [])
    if pairs:
        lines.append("| Cell Pair | Mean Distance (px) | Min Distance (px) | N Pairs |")
        lines.append("|-----------|--------------------|--------------------|---------|")
        for p in pairs:
            lines.append(f"| {p['pair']} | {p['mean_dist']:.1f} | {p['min_dist']:.1f} | {p['n_pairs']} |")
    else:
        lines.append("_Insufficient data for pairwise analysis._")
    lines.append("")
    lines.append("## 5. Spatial Graph Topology")
    lines.append("")
    topo = diag_report.get("graph_topology", {})
    if topo:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in topo.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_Insufficient data for graph analysis._")
    lines.append("")
    lines.append("## 6. Validity Assessment")
    lines.append("")
    lines.append(diag_report.get("validity", "N/A"))
    lines.append("")
    lines.append("## 7. Limitations & Recommendations")
    lines.append("")
    for i, lim in enumerate(diag_report.get("limitations", []), 1):
        lines.append(f"{i}. {lim}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by BioCellAgents \u2014 Multi-Agent Computational Pathology Pipeline*")
    lines.append("*Pipeline: Cellpose Segmentation + DINOv2 ViT-S/14 Classification*")
    lines.append("")
    return "\n".join(lines)



with st.sidebar:
    st.markdown("## BioCellAgents")
    st.markdown("Computational Pathology Analysis")
    st.divider()
    st.markdown("### Settings")
    padding = st.slider("Bounding box padding (px)", 0, 50, 10)
    device = st.selectbox(
        "Compute device", ["cpu", "cuda"],
        index=0 if not torch.cuda.is_available() else 1)
    st.divider()
    with st.expander("About"):
        st.markdown(
            "**Analysis Modules:**\n\n"
            "1. **Cell Segmentation** \u2014 detection quality\n"
            "2. **Cell Classification** \u2014 type distribution\n"
            "3. **Morphometric Analysis** \u2014 spatial patterns\n"
            "4. **Spatial Analysis** \u2014 proximity graph & TME\n"
            "5. **Quality Control** \u2014 statistical reliability\n"
            "6. **Report Synthesis** \u2014 clinical report\n\n"
            "**Pipeline:** Cellpose + DINOv2 ViT-S/14\n"
            "**Dataset:** PanNuke (H&E patches)\n"
            f"**Version:** {APP_VERSION}")

st.markdown(
    """
    <div class="report-header">
        <div class="report-title">BioCellAgents \u2014 Computational Pathology</div>
        <div class="report-subtitle">
            Multi-module analysis pipeline with spatial analytics and tumor microenvironment profiling
        </div>
        <div class="report-meta">
            <span>Pipeline: Cellpose + DINOv2 ViT-S/14</span>
            <span>Dataset: PanNuke (H&E)</span>
            <span>Version: 1.0.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Select Tissue Image")
st.markdown("Choose from **10 sample H&E patches** (Fold 3, PanNuke) or upload your own.")

sample_images = get_sample_images()
gallery_cols = st.columns(5)
for idx, sample in enumerate(sample_images):
    col = gallery_cols[idx % 5]
    with col:
        img_id = sample["id"]
        if st.button(f"Patch #{img_id}", key=f"sel_{img_id}", use_container_width=True):
            st.session_state["selected_patch_id"] = img_id
            st.session_state.pop("uploaded_image", None)
            st.rerun()

selected_image = None
selected_source = ""
if "selected_patch_id" in st.session_state:
    pid = st.session_state["selected_patch_id"]
    sample = next((s for s in sample_images if s["id"] == pid), None)
    if sample:
        selected_image = sample["image"]
        selected_source = f"patch_{pid:04d}.png (Fold 3)"
        st.markdown(f"**Selected:** `{selected_source}` \u2014 {selected_image.size[0]}\u00d7{selected_image.size[1]} px")
        st.image(selected_image, use_container_width=True)

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload H&E image", type=["png", "jpg", "jpeg", "tiff", "bmp"],
    help="PNG, JPG, TIFF, BMP", key="upload_file")
if uploaded_file is not None:
    selected_image = Image.open(uploaded_file).convert("RGB")
    selected_source = uploaded_file.name
    st.session_state.pop("selected_patch_id", None)
    st.markdown(f"**Uploaded:** `{uploaded_file.name}` \u2014 {selected_image.size[0]}\u00d7{selected_image.size[1]} px")
    st.image(selected_image, use_container_width=True)



if selected_image is not None:
    st.markdown("---")
    if st.button("Execute Analysis Pipeline", type="primary", use_container_width=True):
        st.session_state.pop("analysis_results", None)
        progress = st.progress(0, text="Loading models...")
        try:
            analyzer = load_analyzer(device=device)
            progress.progress(20, text="Running Cellpose segmentation...")
            t0 = time.time()
            mask, detected_cells = analyzer.analyze_patch(
                selected_image, CELL_TYPES, padding=padding)
            elapsed = time.time() - t0
            progress.progress(60, text="Running analysis modules + spatial analytics...")
            agent_results, spatial_results, diag_report = run_multi_agent_analysis(
                mask, detected_cells, selected_image.size)
            progress.progress(100, text="Complete.")
            st.session_state["analysis_results"] = {
                "mask": mask, "cells": detected_cells,
                "elapsed": elapsed, "image_size": selected_image.size,
                "source": selected_source, "agents": agent_results,
                "spatial": spatial_results, "diag_report": diag_report,
                "patch_id": st.session_state.get("selected_patch_id"),
            }
            time.sleep(0.3)
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"Analysis failed: {e}")
            st.exception(e)

if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    mask = results["mask"]
    cells = results["cells"]
    elapsed = results["elapsed"]
    image_size = results["image_size"]
    agent_results = results["agents"]
    spatial_results = results["spatial"]
    diag_report = results["diag_report"]

    st.markdown("---")
    st.markdown("## Computational Pathology Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cells Detected", len(cells))
    col2.metric("Inference Time", f"{elapsed:.2f}s")
    col3.metric("Image Size", f"{image_size[0]}\u00d7{image_size[1]}")
    col4.metric("Foreground", f"{100*np.mean(mask>0):.1f}%" if mask.size > 0 else "N/A")

    tab_report, tab_vis, tab_charts, tab_explorer, tab_table, tab_export = st.tabs([
        "Clinical Report",
        "Visual Analysis",
        "Charts & Spatial",
        "Cell Explorer",
        "Cell Table",
        "Export",
    ])

    with tab_report:
        st.markdown("### Analysis Findings")
        st.markdown(
            "Each module independently analyzes the image. The Spatial Analysis module "
            "computes proximity graphs and TME metrics. The Report Synthesis module "
            "integrates all results into a structured clinical report.")
        st.divider()

        for agent_name, findings in agent_results.items():
            if agent_name in ("Report Synthesis", "Spatial Analysis"):
                continue
            st.markdown(f"**{agent_name}**")
            for f in findings:
                sev_class = f.severity if f.severity != "info" else ""
                st.markdown(
                    f'<div class="finding-block {sev_class}">{f.text}</div>',
                    unsafe_allow_html=True)
            st.markdown("")

        st.markdown("**Spatial Analysis**")
        if spatial_results.get("metrics"):
            m = spatial_results["metrics"]
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Graph Density", f"{m['density']:.4f}")
            sm2.metric("Avg Degree", f"{m['avg_degree']:.2f}")
            sm3.metric("Clustering Coeff.", f"{m['clustering_coeff']:.4f}")
            sm4.metric("Edges (<50px)", m["n_edges"])
        if spatial_results.get("tme", {}).get("inflammatory_neoplastic_mean") is not None:
            tme = spatial_results["tme"]
            st.markdown(f"""
            <div class="finding-block success">
                TME: Inflammatory <-> Neoplastic mean distance = {tme['inflammatory_neoplastic_mean']:.1f}px
                (range: {tme['inflammatory_neoplastic_min']:.1f}-{tme['inflammatory_neoplastic_max']:.1f}px,
                n={tme['inflammatory_neoplastic_count']} pairs)
            </div>""", unsafe_allow_html=True)



        st.markdown("**Report Synthesis**")
        st.markdown(f"""
        <div class="report-section">
            <h4>Executive Summary</h4>
            <p>{diag_report.get('executive_summary', 'N/A')}</p>
        </div>""", unsafe_allow_html=True)

        if diag_report.get("cellular_breakdown"):
            st.markdown("#### Cellular Breakdown")
            cb_df = pd.DataFrame(diag_report["cellular_breakdown"])
            st.dataframe(cb_df, use_container_width=True, hide_index=True)

        # Ground Truth Comparison (trial mode only)
        patch_id = results.get("patch_id")
        if patch_id is not None:
            gt_cells = load_ground_truth(patch_id)
            if gt_cells:
                st.markdown("#### Classification Validation (vs. Ground Truth)")
                st.markdown(
                    f"Comparing model predictions against PanNuke ground truth annotations "
                    f"({len(gt_cells)} annotated cells in this patch).")
                comparison = compare_predictions(cells, gt_cells)
                if comparison:
                    comp_df = pd.DataFrame(comparison["comparison"])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Ground Truth Cells", comparison["total_gt"])
                    m2.metric("Predicted Cells", comparison["total_pred"])
                    m3.metric("Proportion Agreement", f"{comparison['agreement']:.1%}")
                    if comparison["agreement"] >= 0.8:
                        st.markdown(
                            '<div class="finding-block success">Classification agreement with ground truth is strong.</div>',
                            unsafe_allow_html=True)
                    elif comparison["agreement"] >= 0.5:
                        st.markdown(
                            '<div class="finding-block warning">Moderate agreement with ground truth. Some misclassifications detected.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="finding-block error">Low agreement with ground truth. Model may need recalibration.</div>',
                            unsafe_allow_html=True)

        if diag_report.get("tme_metrics"):
            st.markdown("#### TME Metrics")
            tme_df = pd.DataFrame(list(diag_report["tme_metrics"].items()), columns=["Metric", "Value"])
            st.dataframe(tme_df, use_container_width=True, hide_index=True)

        if diag_report.get("pairwise_matrix"):
            st.markdown("#### Pairwise Interaction Matrix")
            pw_df = pd.DataFrame(diag_report["pairwise_matrix"])
            st.dataframe(pw_df, use_container_width=True, hide_index=True)

        if diag_report.get("graph_topology"):
            st.markdown("#### Spatial Graph Topology")
            gt_df = pd.DataFrame(list(diag_report["graph_topology"].items()), columns=["Metric", "Value"])
            st.dataframe(gt_df, use_container_width=True, hide_index=True)

        st.markdown("#### Quality Control")
        st.markdown(f"`{diag_report.get('validity', 'N/A')}`")

        st.markdown("#### Limitations & Recommendations")
        for i, lim in enumerate(diag_report.get("limitations", []), 1):
            st.markdown(f"{i}. {lim}")

    with tab_vis:
        st.markdown("### Visual Analysis")
        
        # --- Overlay Controls ---
        st.markdown("#### Overlay Controls")
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            overlay_opacity = st.slider(
                "Overlay Opacity",
                min_value=0.1,
                max_value=1.0,
                value=0.55,
                step=0.05,
                key="overlay_opacity",
            )
        with ctrl2:
            show_connections = st.checkbox(
                "Show Connections",
                value=True,
                key="show_connections",
            )
        with ctrl3:
            highlight_mode = st.selectbox(
                "Highlight Mode",
                options=["None", "Low Confidence", "High Confidence"],
                index=0,
                key="highlight_mode",
            )
        
        vis1, vis2 = st.columns(2)
        with vis1:
            st.markdown("**Original Image**")
            st.image(selected_image, use_container_width=True)
        with vis2:
            st.markdown("**Cell Annotations**")
            annotated = annotate_image(selected_image, cells)
            st.image(annotated, use_container_width=True)
        
        st.markdown("**Segmentation Overlay**")
        overlay = create_overlay(selected_image, mask, cells, alpha=overlay_opacity)
        st.image(overlay, use_container_width=True)
        
        st.markdown("**Segmentation Mask**")
        st.image(mask_to_display(mask), use_container_width=True)

    with tab_charts:
        st.markdown("### Charts & Spatial Analytics")
        if cells:
            # --- Basic Distribution Charts ---
            pie, bar = build_charts(cells)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(pie, use_container_width=True)
            with ch2:
                st.plotly_chart(bar, use_container_width=True)
            
            # --- Confidence & Size Histograms ---
            st.markdown("---")
            st.markdown("#### Distribution Analysis")
            hist1, hist2 = st.columns(2)
            with hist1:
                conf_hist = build_confidence_histogram(cells)
                st.plotly_chart(conf_hist, use_container_width=True)
            with hist2:
                size_hist = build_size_histogram(cells)
                st.plotly_chart(size_hist, use_container_width=True)
            
            # --- Spatial Scatter ---
            st.markdown("---")
            spatial_fig = build_spatial_scatter(cells, spatial_results)
            if spatial_fig:
                st.plotly_chart(spatial_fig, use_container_width=True)
            
            # --- Interaction Heatmap ---
            st.markdown("---")
            heatmap_fig = build_interaction_heatmap(spatial_results)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # --- Network Graph ---
            st.markdown("---")
            network_fig = build_network_graph(spatial_results, cells)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("No cells detected for charting.")

    with tab_explorer:
        render_cell_explorer(cells)

    with tab_table:
        st.markdown("### Cell Detection Table")
        if cells:
            df = build_dataframe(cells)
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"{len(cells)} cells detected")
        else:
            st.info("No cells detected.")

    with tab_export:
        st.markdown("### Export Results")
        if cells:
            json_data = export_json(cells, image_size, source=results["source"],
                                   diag_report=diag_report, spatial_results=spatial_results)
            csv_data = export_csv(cells)
            md_report = generate_markdown_report(
                diag_report, spatial_results, cells, image_size, results["source"])
            pdf_data = generate_pdf_report(
                diag_report, spatial_results, cells, image_size, results["source"])
            ex1, ex2, ex3, ex4 = st.columns(4)
            with ex1:
                st.download_button("Download JSON", json_data,
                                  "biocell_results.json", "application/json", use_container_width=True)
            with ex2:
                st.download_button("Download CSV", csv_data,
                                  "biocell_cells.csv", "text/csv", use_container_width=True)
            with ex3:
                st.download_button("Download Report (MD)", md_report,
                                  "biocell_report.md", "text/markdown", use_container_width=True)
            with ex4:
                st.download_button("Download Report (PDF)", pdf_data,
                                  "biocell_report.pdf", "application/pdf", use_container_width=True)
            st.markdown("---")
            st.markdown("#### Full Report Preview")
            st.markdown(md_report)
        else:
            st.info("No data to export.")

