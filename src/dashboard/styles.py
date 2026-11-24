"""
BioCellAgents Dashboard Styles — Clinical Dark Precision UI
===========================================================
Unified color palettes (clinical + Okabe-Ito colorblind-safe) and dark
glassmorphic CSS aligned with ``.streamlit/config.toml``.
"""

from __future__ import annotations

from typing import Dict

# Primary cell type colors (clinical high-contrast dark-theme palette)
CELL_COLORS: Dict[str, str] = {
    "Neoplastic": "#F43F5E",
    "Inflammatory": "#F97316",
    "Connective": "#38BDF8",
    "Dead": "#A78BFA",
    "Epithelial": "#10B981",
    "Unknown": "#94A3B8",
}

# Okabe-Ito color-blind-friendly palette
OKABE_ITO_COLORS: Dict[str, str] = {
    "Neoplastic": "#D55E00",
    "Inflammatory": "#E69F00",
    "Connective": "#56B4E9",
    "Dead": "#CC79A7",
    "Epithelial": "#009E73",
    "Unknown": "#94A3B8",
}

# Accent colors (matching .streamlit/config.toml)
ACCENT = "#4FC3F7"
BG_DARK = "#0E1117"
SURFACE_DARK = "#1A1F2E"
TEXT_PRIMARY = "#E2E8F0"
TEXT_MUTED = "#94A3B8"


def get_palette(colorblind_safe: bool = False) -> Dict[str, str]:
    """Return the active 5-class cell color mapping."""
    return dict(OKABE_ITO_COLORS if colorblind_safe else CELL_COLORS)


def get_custom_css() -> str:
    """Return the Clinical Dark Precision CSS string for Streamlit injection."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* --- Header Hero Banner --- */
    .biocell-hero {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.92), rgba(14, 17, 23, 0.95));
        border: 1px solid rgba(79, 195, 247, 0.25);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    }
    .biocell-hero-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
    }
    .biocell-hero-sub {
        font-size: 0.9rem;
        color: #94A3B8;
        margin: 0;
    }

    /* --- Hardware & Status Badges --- */
    .hw-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.25rem 0 0.65rem 0;
    }
    .hw-badge-gpu {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34D399;
    }
    .hw-badge-cpu {
        background: rgba(79, 195, 247, 0.12);
        border: 1px solid rgba(79, 195, 247, 0.38);
        color: #7DD3FC;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-pass {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34D399;
    }
    .status-warn {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #FBBF24;
    }
    .status-fail {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.45);
        color: #FB7185;
    }

    /* --- Glassmorphic KPI Metric Cards --- */
    .metric-card {
        background: rgba(26, 31, 46, 0.72);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.65rem;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(79, 195, 247, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #7DD3FC;
        margin-top: 0.18rem;
    }

    /* --- Agent Pipeline Stepper --- */
    .pipeline-step {
        background: rgba(26, 31, 46, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        text-align: center;
    }
    .pipeline-step-done {
        border-color: rgba(16, 185, 129, 0.55);
        background: rgba(16, 185, 129, 0.10);
    }
    .pipeline-step-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #F1F5F9;
    }
    .pipeline-step-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.15rem;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(26, 31, 46, 0.65);
        padding: 5px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        color: #CBD5E1;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 195, 247, 0.16) !important;
        color: #4FC3F7 !important;
        font-weight: 600;
    }

    /* --- Sidebar & Scrollbars --- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.15); }
    ::-webkit-scrollbar-thumb { background: rgba(79, 195, 247, 0.35); border-radius: 4px; }
    </style>
    """
