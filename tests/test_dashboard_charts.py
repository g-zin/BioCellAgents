"""
Unit tests for BioCellAgents dashboard chart builders.
"""

import pytest
import numpy as np
import plotly.graph_objects as go

from src.dashboard.charts import (
    build_confidence_histogram,
    build_size_histogram,
    build_interaction_heatmap,
    build_network_graph,
)
from src.dashboard.styles import CELL_COLORS, OKABE_ITO_COLORS


# --- Test fixtures ---

@pytest.fixture
def sample_cells():
    """Create a sample list of cell dicts for testing."""
    return [
        {
            "cell_id": "cell_001",
            "class_name": "Neoplastic",
            "confidence": 0.95,
            "bbox": [10, 20, 50, 60],
            "centroid": (30, 40),
        },
        {
            "cell_id": "cell_002",
            "class_name": "Inflammatory",
            "confidence": 0.85,
            "bbox": [70, 80, 110, 120],
            "centroid": (90, 100),
        },
        {
            "cell_id": "cell_003",
            "class_name": "Connective",
            "confidence": 0.75,
            "bbox": [130, 140, 170, 180],
            "centroid": (150, 160),
        },
        {
            "cell_id": "cell_004",
            "class_name": "Dead",
            "confidence": 0.65,
            "bbox": [190, 200, 230, 240],
            "centroid": (210, 220),
        },
        {
            "cell_id": "cell_005",
            "class_name": "Epithelial",
            "confidence": 0.55,
            "bbox": [250, 260, 290, 300],
            "centroid": (270, 280),
        },
    ]


@pytest.fixture
def sample_spatial_results():
    """Create a sample spatial results dict for testing."""
    return {
        "graph": np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ]),
        "type_pairs": [
            {"pair": "Neoplastic <-> Inflammatory", "mean_dist": 45.2, "min_dist": 30.1, "n_pairs": 10},
            {"pair": "Inflammatory <-> Connective", "mean_dist": 52.8, "min_dist": 35.4, "n_pairs": 8},
            {"pair": "Connective <-> Dead", "mean_dist": 48.3, "min_dist": 32.7, "n_pairs": 12},
            {"pair": "Dead <-> Epithelial", "mean_dist": 55.1, "min_dist": 40.2, "n_pairs": 6},
        ],
        "metrics": {
            "density": 0.2,
            "avg_degree": 1.2,
            "clustering_coeff": 0.3,
            "n_nodes": 5,
            "n_edges": 4,
        },
    }


# --- Test styles ---

def test_cell_colors_defined():
    """Test that CELL_COLORS is properly defined."""
    assert "Neoplastic" in CELL_COLORS
    assert "Inflammatory" in CELL_COLORS
    assert "Connective" in CELL_COLORS
    assert "Dead" in CELL_COLORS
    assert "Epithelial" in CELL_COLORS


def test_okabe_ito_colors_defined():
    """Test that OKABE_ITO_COLORS is properly defined."""
    assert "Neoplastic" in OKABE_ITO_COLORS
    assert "Inflammatory" in OKABE_ITO_COLORS
    assert "Connective" in OKABE_ITO_COLORS
    assert "Dead" in OKABE_ITO_COLORS
    assert "Epithelial" in OKABE_ITO_COLORS


# --- Test chart builders ---

def test_build_confidence_histogram(sample_cells):
    """Test that build_confidence_histogram returns a valid Plotly figure."""
    fig = build_confidence_histogram(sample_cells)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.layout.title.text == "Confidence Distribution by Cell Type"


def test_build_size_histogram(sample_cells):
    """Test that build_size_histogram returns a valid Plotly figure."""
    fig = build_size_histogram(sample_cells)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert "Cell Size Distribution" in fig.layout.title.text


def test_build_interaction_heatmap(sample_spatial_results):
    """Test that build_interaction_heatmap returns a valid Plotly figure."""
    fig = build_interaction_heatmap(sample_spatial_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert "Interaction Matrix" in fig.layout.title.text


def test_build_interaction_heatmap_empty():
    """Test that build_interaction_heatmap returns None for empty data."""
    assert build_interaction_heatmap({}) is None
    assert build_interaction_heatmap({"type_pairs": []}) is None


def test_build_network_graph(sample_spatial_results, sample_cells):
    """Test that build_network_graph returns a valid Plotly figure."""
    fig = build_network_graph(sample_spatial_results, sample_cells)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert "Spatial Proximity Network" in fig.layout.title.text


def test_build_network_graph_no_graph():
    """Test that build_network_graph returns None when no graph data."""
    assert build_network_graph({}, sample_cells) is None
    assert build_network_graph({"graph": None}, sample_cells) is None


def test_build_network_graph_single_cell():
    """Test that build_network_graph returns None for single cell."""
    single_cell = [{"cell_id": "cell_001", "class_name": "Neoplastic", "confidence": 0.9}]
    spatial = {"graph": np.array([[0]])}
    assert build_network_graph(spatial, single_cell) is None
