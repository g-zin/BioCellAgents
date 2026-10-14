"""
BioCellAgents Dashboard Charts
==============================
Interactive chart builders for the BioCellAgents dashboard.
"""

import numpy as np
import plotly.graph_objects as go

from src.data.dataset import CELL_TYPES
from src.dashboard.styles import CELL_COLORS


def build_confidence_histogram(cells: list) -> go.Figure:
    """Build a confidence distribution histogram per cell class.

    Parameters:
        cells: List of cell dicts with 'class_name' and 'confidence' keys.

    Returns:
        go.Figure: Plotly histogram figure.
    """
    fig = go.Figure()
    for ct in CELL_TYPES:
        subset = [c for c in cells if c["class_name"] == ct]
        if not subset:
            continue
        confs = [c["confidence"] for c in subset]
        fig.add_trace(go.Histogram(
            x=confs,
            name=ct,
            marker=dict(color=CELL_COLORS.get(ct, "#888")),
            opacity=0.7,
            nbinsx=20,
        ))
    fig.update_layout(
        title="Confidence Distribution by Cell Type",
        barmode="overlay",
        xaxis=dict(title="Confidence Score", range=[0, 1]),
        yaxis=dict(title="Count"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13),
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def build_size_histogram(cells: list) -> go.Figure:
    """Build a cell size (area) distribution histogram per cell class.

    Parameters:
        cells: List of cell dicts with 'class_name' and 'bbox' keys.

    Returns:
        go.Figure: Plotly histogram figure.
    """
    fig = go.Figure()
    for ct in CELL_TYPES:
        subset = [c for c in cells if c["class_name"] == ct]
        if not subset:
            continue
        areas = []
        for c in subset:
            bbox = c["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            areas.append(w * h)
        fig.add_trace(go.Histogram(
            x=areas,
            name=ct,
            marker=dict(color=CELL_COLORS.get(ct, "#888")),
            opacity=0.7,
            nbinsx=20,
        ))
    fig.update_layout(
        title="Cell Size Distribution (Area in px\u00b2)",
        barmode="overlay",
        xaxis=dict(title="Area (px\u00b2)", type="log"),
        yaxis=dict(title="Count"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13),
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def build_interaction_heatmap(spatial_results: dict) -> go.Figure | None:
    """Build a heatmap of the pairwise cell-cell interaction matrix.

    Parameters:
        spatial_results: Dict with 'type_pairs' key (list of pair dicts).

    Returns:
        go.Figure or None: Plotly heatmap figure, or None if no data.
    """
    if not spatial_results or "type_pairs" not in spatial_results:
        return None

    type_pairs = spatial_results["type_pairs"]
    if not type_pairs:
        return None

    # Build a matrix from type_pairs
    cell_types = list(CELL_TYPES)
    n = len(cell_types)
    data = np.zeros((n, n), dtype=float)

    for tp in type_pairs:
        pair = tp["pair"]  # e.g. "Neoplastic <-> Inflammatory"
        parts = pair.split("<->")
        if len(parts) != 2:
            continue
        t1 = parts[0].strip()
        t2 = parts[1].strip()
        if t1 in cell_types and t2 in cell_types:
            i = cell_types.index(t1)
            j = cell_types.index(t2)
            data[i, j] = tp["mean_dist"]
            data[j, i] = tp["mean_dist"]

    # Check if there's any data
    if data.max() == 0:
        return None

    fig = go.Figure(go.Heatmap(
        z=data,
        x=cell_types,
        y=cell_types,
        colorscale="Viridis",
        text=data.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="Mean Dist (px)"),
    ))
    fig.update_layout(
        title="Cell-Cell Interaction Matrix (Mean Distance)",
        xaxis=dict(title="Target Cell Type", side="bottom"),
        yaxis=dict(title="Source Cell Type", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13),
        height=450,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def build_network_graph(spatial_results: dict, cells: list) -> go.Figure | None:
    """Build an interactive network graph visualization.

    Parameters:
        spatial_results: Dict with 'graph' key (adjacency matrix).
        cells: List of cell dicts.

    Returns:
        go.Figure or None: Plotly network figure, or None if no data.
    """
    if not spatial_results or spatial_results.get("graph") is None:
        return None

    adj = spatial_results["graph"]
    n = len(cells)
    if n < 2:
        return None

    # Compute spring layout positions
    import networkx as nx
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, k=2.0, iterations=50)

    # Build edge traces
    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edge_x.extend([pos[i][0], pos[j][0], None])
                edge_y.extend([pos[i][1], pos[j][1], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(100,140,255,0.3)"),
        name="edges",
        showlegend=False,
    ))

    # Node traces per class
    for ct in CELL_TYPES:
        subset = [(i, c) for i, c in enumerate(cells) if c["class_name"] == ct]
        if not subset:
            continue
        xs = [pos[i][0] for i, _ in subset]
        ys = [pos[i][1] for i, _ in subset]
        sizes = [8 + 2 * adj[i].sum() for i, _ in subset]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=ct,
            marker=dict(
                size=sizes,
                color=CELL_COLORS.get(ct, "#888"),
                line=dict(width=1, color="white"),
            ),
            text=[f"{c['cell_id']} ({c['confidence']:.0%})" for _, c in subset],
        ))

    fig.update_layout(
        title="Spatial Proximity Network",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13),
        height=500,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
