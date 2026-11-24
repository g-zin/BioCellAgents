"""
BioCellAgents Dashboard Charts — Clinical Dark Precision Plotly Builders
========================================================================
Unified dark-mode-ready interactive Plotly chart builders shared by
``biocell_app.py`` and ``src/dashboard/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from src.dashboard.styles import ACCENT, CELL_COLORS, TEXT_PRIMARY, get_palette
from src.data.dataset import CELL_TYPES

_DARK_GRID = "rgba(148, 163, 184, 0.15)"


def _apply_dark_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_PRIMARY, size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=12),
        height=height,
        margin=dict(t=48, b=28, l=36, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRIMARY)),
    )
    return fig


def build_cell_pie(
    counts: Dict[str, int],
    palette: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Build a 5-class PanNuke cellular composition donut chart."""
    colors = palette or CELL_COLORS
    labels = [k for k, v in counts.items() if int(v or 0) > 0]
    values = [int(counts[k]) for k in labels]
    marker_colors = [colors.get(k, "#94A3B8") for k in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            marker=dict(colors=marker_colors, line=dict(color="#0E1117", width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    return _apply_dark_layout(fig, "5-Class PanNuke Cell Composition", height=340)


def build_organ_bar(top3: List[Dict[str, Any]]) -> go.Figure:
    """Build a horizontal calibrated confidence bar chart for top-k organs."""
    organs = [str(item.get("organ", "Unknown")) for item in top3]
    confs = [float(item.get("confidence", 0.0) or 0.0) for item in top3]

    fig = go.Figure(
        go.Bar(
            x=confs,
            y=organs,
            orientation="h",
            marker=dict(
                color=[
                    ACCENT if i == 0 else "rgba(79, 195, 247, 0.45)" for i in range(len(organs))
                ],
                line=dict(color="rgba(79, 195, 247, 0.8)", width=1),
            ),
            text=[f"{c:.1%}" for c in confs],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Calibrated Confidence: %{x:.4f}<extra></extra>",
        )
    )
    _apply_dark_layout(fig, "Top-K Organ Predictions (Calibrated)", height=280)
    fig.update_xaxes(range=[0, 1.05], gridcolor=_DARK_GRID, title="Probability")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def build_spatial_scatter(
    centroids: List[Dict[str, Any]],
    palette: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Build an interactive 2D spatial cell-centroid scatter plot."""
    colors = palette or CELL_COLORS
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pt in centroids:
        cls = str(pt.get("class") or pt.get("class_name") or "Unknown")
        grouped.setdefault(cls, []).append(pt)

    fig = go.Figure()
    for cls, pts in grouped.items():
        xs = [float(p.get("x", 0.0) or 0.0) for p in pts]
        ys = [float(p.get("y", 0.0) or 0.0) for p in pts]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"{cls} ({len(pts)})",
                marker=dict(
                    size=9,
                    color=colors.get(cls, "#94A3B8"),
                    line=dict(color="#0E1117", width=1),
                ),
                hovertemplate=f"<b>{cls}</b><br>x=%{{x:.1f}} px, y=%{{y:.1f}} px<extra></extra>",
            )
        )
    _apply_dark_layout(fig, "2D Spatial Nuclei Map", height=400)
    fig.update_xaxes(title="X (px)", gridcolor=_DARK_GRID)
    fig.update_yaxes(title="Y (px)", autorange="reversed", gridcolor=_DARK_GRID)
    return fig


def build_interaction_count_heatmap(interaction_matrix: Dict[str, int]) -> Optional[go.Figure]:
    """Build a 5x5 heatmap from pairwise cell-class edge count dictionary."""
    if not interaction_matrix:
        return None
    cell_types = list(CELL_TYPES)
    n = len(cell_types)
    mat = np.zeros((n, n), dtype=int)
    for pair_key, count in interaction_matrix.items():
        sep = "<->" if "<->" in pair_key else "-"
        parts = [p.strip() for p in pair_key.split(sep)]
        if len(parts) == 2 and parts[0] in cell_types and parts[1] in cell_types:
            i, j = cell_types.index(parts[0]), cell_types.index(parts[1])
            mat[i, j] = int(count)
            mat[j, i] = int(count)

    if int(mat.max()) == 0:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=mat,
            x=cell_types,
            y=cell_types,
            colorscale="Tealgrn",
            text=mat.astype(str),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#0E1117"),
            colorbar=dict(title="Edges"),
        )
    )
    _apply_dark_layout(fig, "Pairwise Cell-Cell Interaction Matrix (Edge Counts)", height=390)
    fig.update_yaxes(autorange="reversed")
    return fig


def build_confidence_histogram(
    cells: list,
    colorblind_safe: bool = False,
) -> go.Figure:
    """Build a confidence distribution histogram per cell class."""
    palette = get_palette(colorblind_safe)
    fig = go.Figure()
    for ct in CELL_TYPES:
        subset = [c for c in cells if c["class_name"] == ct]
        if not subset:
            continue
        confs = [c["confidence"] for c in subset]
        fig.add_trace(
            go.Histogram(
                x=confs,
                name=ct,
                marker=dict(color=palette.get(ct, "#94A3B8")),
                opacity=0.75,
                nbinsx=20,
            )
        )
    _apply_dark_layout(fig, "Confidence Distribution by Cell Type", height=380)
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Confidence Score", range=[0, 1], gridcolor=_DARK_GRID),
        yaxis=dict(title="Count", gridcolor=_DARK_GRID),
    )
    return fig


def build_size_histogram(
    cells: list,
    colorblind_safe: bool = False,
) -> go.Figure:
    """Build a cell size (area) distribution histogram per cell class."""
    palette = get_palette(colorblind_safe)
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
        fig.add_trace(
            go.Histogram(
                x=areas,
                name=ct,
                marker=dict(color=palette.get(ct, "#94A3B8")),
                opacity=0.75,
                nbinsx=20,
            )
        )
    _apply_dark_layout(fig, "Cell Size Distribution (Area in px\u00b2)", height=380)
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Area (px\u00b2)", type="log", gridcolor=_DARK_GRID),
        yaxis=dict(title="Count", gridcolor=_DARK_GRID),
    )
    return fig


def build_interaction_heatmap(spatial_results: dict) -> go.Figure | None:
    """Build a heatmap of the pairwise cell-cell interaction matrix."""
    if not spatial_results or "type_pairs" not in spatial_results:
        return None

    type_pairs = spatial_results["type_pairs"]
    if not type_pairs:
        return None

    cell_types = list(CELL_TYPES)
    n = len(cell_types)
    data = np.zeros((n, n), dtype=float)

    for tp in type_pairs:
        pair = tp["pair"]
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

    if data.max() == 0:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=data,
            x=cell_types,
            y=cell_types,
            colorscale="Viridis",
            text=data.astype(str),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="Mean Dist (px)"),
        )
    )
    _apply_dark_layout(fig, "Cell-Cell Interaction Matrix (Mean Distance)", height=430)
    fig.update_xaxes(title="Target Cell Type", side="bottom")
    fig.update_yaxes(title="Source Cell Type", autorange="reversed")
    return fig


def build_network_graph(
    spatial_results: dict,
    cells: list,
    colorblind_safe: bool = False,
) -> go.Figure | None:
    """Build an interactive 2D spatial proximity network graph."""
    if not spatial_results or spatial_results.get("graph") is None:
        return None

    adj = spatial_results["graph"]
    if adj.shape[0] < 2 or len(cells) < 2:
        return None

    palette = get_palette(colorblind_safe)
    n = min(adj.shape[0], len(cells))

    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for i in range(n):
        c1 = cells[i].get("centroid", (0, 0))
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                c2 = cells[j].get("centroid", (0, 0))
                edge_x.extend([float(c1[0]), float(c2[0]), None])
                edge_y.extend([float(c1[1]), float(c2[1]), None])

    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.0, color="rgba(148, 163, 184, 0.35)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    for ct in CELL_TYPES:
        xs, ys = [], []
        for idx in range(n):
            if cells[idx].get("class_name") == ct:
                cx, cy = cells[idx].get("centroid", (0, 0))
                xs.append(float(cx))
                ys.append(float(cy))
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=ct,
                    marker=dict(
                        size=9,
                        color=palette.get(ct, "#94A3B8"),
                        line=dict(color="#0E1117", width=1),
                    ),
                )
            )

    _apply_dark_layout(fig, "Spatial Proximity Network", height=430)
    fig.update_xaxes(title="X (px)", gridcolor=_DARK_GRID)
    fig.update_yaxes(title="Y (px)", autorange="reversed", gridcolor=_DARK_GRID)
    return fig
