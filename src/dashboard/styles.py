"""
BioCellAgents Dashboard Styles
==============================
Color palettes and CSS definitions for the interactive dashboard.
"""

# Primary cell type colors (clinical palette)
CELL_COLORS = {
    "Neoplastic": "#DC2626",
    "Inflammatory": "#EA580C",
    "Connective": "#2563EB",
    "Dead": "#7C3AED",
    "Epithelial": "#059669",
}

# Okabe-Ito color-blind-friendly palette (alternative)
OKABE_ITO_COLORS = {
    "Neoplastic": "#D55E00",
    "Inflammatory": "#E69F00",
    "Connective": "#0072B2",
    "Dead": "#CC79A7",
    "Epithelial": "#009E73",
}

# Accent color (matches Streamlit primaryColor)
ACCENT = "#4FC3F7"


def get_custom_css() -> str:
    """Return the custom CSS string for the dashboard.

    Returns:
        str: CSS string to be injected via st.markdown(unsafe_allow_html=True)
    """
    return """
    /* --- Card containers --- */
    .stContainer {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .stContainer:hover {
        border-color: rgba(79, 195, 247, 0.3);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* --- Custom scrollbar --- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(79, 195, 247, 0.3); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(79, 195, 247, 0.5); }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(0, 0, 0, 0.2);
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 195, 247, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 195, 247, 0.15) !important;
        color: #4FC3F7 !important;
    }

    /* --- Metrics --- */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(79, 195, 247, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* --- Buttons --- */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.2);
    }

    /* --- Data editor --- */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* --- Responsive --- */
    @media (max-width: 768px) {
        .stContainer { padding: 0.75rem; }
        div[data-testid="stMetric"] { padding: 0.75rem; }
    }
    """
