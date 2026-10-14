"""
BioCellAgents Dashboard Explorer
================================
Interactive cell explorer with filters for the BioCellAgents dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.dataset import CELL_TYPES
from src.dashboard.styles import CELL_COLORS


def render_cell_explorer(cells: list) -> None:
    """Render the interactive cell explorer tab.

    Parameters:
        cells: List of cell dicts with 'cell_id', 'class_name', 'confidence',
               'bbox', 'centroid' keys.
    """
    if not cells:
        st.info("No cells detected for exploration.")
        return

    st.markdown("### Cell Explorer")
    st.markdown(
        "Filter and explore individual cell detections. "
        "Use the filters below to narrow down the results."
    )

    # --- Filters ---
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_classes = st.multiselect(
            "Cell Class",
            options=CELL_TYPES,
            default=CELL_TYPES,
            key="explorer_class_filter",
        )

    with filter_col2:
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="explorer_conf_filter",
        )

    with filter_col3:
        min_area = st.slider(
            "Min Area (px\u00b2)",
            min_value=0,
            max_value=10000,
            value=0,
            step=100,
            key="explorer_area_filter",
        )

    # --- Apply filters ---
    filtered_cells = []
    for cell in cells:
        if cell["class_name"] not in selected_classes:
            continue
        if cell["confidence"] < min_confidence:
            continue
        bbox = cell["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue
        filtered_cells.append(cell)

    # --- Summary metrics ---
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cells", len(cells))
    m2.metric("Filtered", len(filtered_cells))
    m3.metric("Classes", len(selected_classes))
    m4.metric("Min Conf", f"{min_confidence:.2f}")

    if not filtered_cells:
        st.warning("No cells match the current filters. Try adjusting the criteria.")
        return

    # --- Filtered table ---
    st.markdown("#### Filtered Cells")
    rows = []
    for cell in filtered_cells:
        bbox = cell["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        rows.append({
            "ID": cell["cell_id"],
            "Class": cell["class_name"],
            "Confidence": f"{cell['confidence']:.3f}",
            "Area (px\u00b2)": int(area),
            "Centroid X": int(cell["centroid"][0]),
            "Centroid Y": int(cell["centroid"][1]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400, hide_index=True)

    # --- Filtered scatter ---
    st.markdown("#### Spatial Distribution (Filtered)")
    fig = go.Figure()
    for ct in CELL_TYPES:
        subset = [c for c in filtered_cells if c["class_name"] == ct]
        if not subset:
            continue
        xs = [c["centroid"][0] for c in subset]
        ys = [c["centroid"][1] for c in subset]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=ct,
            marker=dict(size=8, color=CELL_COLORS.get(ct, "#888")),
            text=[f"{c['cell_id']} ({c['confidence']:.0%})" for c in subset],
        ))
    fig.update_layout(
        title="Filtered Cell Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", size=13),
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(title="X (px)", gridcolor="#cbd5e1"),
        yaxis=dict(title="Y (px)", gridcolor="#cbd5e1", scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig, use_container_width=True)
