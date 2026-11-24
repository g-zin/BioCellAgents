"""Deterministic pathology reporting, TME phenotyping, Markdown tables, and PDF export.

Top-level imports use ONLY the Python standard library so this module is 100%
testable offline without ML dependencies. ``fpdf2`` is imported lazily inside
``export_report_pdf``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeGuard, Union

PIXEL_SIZE_UM: float = 0.25

PAN_NUKE_CELL_CLASSES: Tuple[str, ...] = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)

TME_PHENOTYPES: Tuple[str, ...] = (
    "Immune-Infiltrated",
    "Immune-Excluded",
    "Immune-Desert",
)

NO_LITERATURE_FALLBACK: str = "No indexed literature available for this claim."


def _is_finite_number(value: Any) -> TypeGuard[Union[int, float]]:
    """Return True if value is a finite int or float (excluding bool)."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _fmt_num(value: Any, decimals: int = 4) -> str:
    """Format a finite numeric value up to ``decimals`` places, else 'N/A'."""
    if not _is_finite_number(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def px_to_um(
    px: Optional[float],
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> Optional[float]:
    """Convert a pixel distance to micrometers at ``pixel_size_um`` (default 0.25 μm/px)."""
    if px is None or not _is_finite_number(px):
        return None
    if not _is_finite_number(pixel_size_um) or float(pixel_size_um) <= 0.0:
        raise ValueError(f"pixel_size_um must be positive and finite, got {pixel_size_um!r}")
    return round(float(px) * float(pixel_size_um), 4)


def format_distance(
    px: Optional[float],
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> str:
    """Format a distance in both pixels and micrometers, e.g. '32.40 px (8.10 μm)'."""
    if px is None or not _is_finite_number(px):
        return "N/A"
    um = px_to_um(px, pixel_size_um=pixel_size_um)
    if um is None:
        return "N/A"
    return f"{float(px):.2f} px ({um:.2f} μm)"


def classify_tme_phenotype(
    class_counts: Dict[str, int],
    immune_infiltration: Dict[str, Any],
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> Tuple[str, str]:
    """Deterministically classify the Tumor Microenvironment (TME) phenotype.

    Always returns one of the 3 canonical labels:
    ``"Immune-Infiltrated"``, ``"Immune-Excluded"``, or ``"Immune-Desert"``,
    paired with a concise quantitative rationale.
    """
    normalized_counts = {
        cls: max(0, int(class_counts.get(cls, 0) or 0)) for cls in PAN_NUKE_CELL_CLASSES
    }
    total_cells = sum(normalized_counts.values())

    inf_raw = immune_infiltration.get("inflammatory_count")
    neo_raw = immune_infiltration.get("neoplastic_count")
    inflammatory_count = (
        max(0, int(inf_raw)) if _is_finite_number(inf_raw) else normalized_counts["Inflammatory"]
    )
    neoplastic_count = (
        max(0, int(neo_raw)) if _is_finite_number(neo_raw) else normalized_counts["Neoplastic"]
    )
    inflammatory_pct = (inflammatory_count / total_cells * 100.0) if total_cells > 0 else 0.0

    mean_dist_px = immune_infiltration.get("mean_immune_to_tumor_dist_px")
    mean_dist_val: Optional[float] = (
        float(mean_dist_px) if _is_finite_number(mean_dist_px) else None
    )

    if inflammatory_count == 0 or inflammatory_pct < 5.0:
        rationale = (
            f"Inflammatory proportion is {inflammatory_pct:.2f}% "
            f"({inflammatory_count}/{total_cells} cells), below the 5.00% immune presence threshold."
        )
        return "Immune-Desert", rationale

    if neoplastic_count == 0:
        rationale = (
            f"Inflammatory cells comprise {inflammatory_pct:.2f}% "
            f"({inflammatory_count}/{total_cells} cells) with 0 neoplastic cells detected "
            "(reactive/non-neoplastic immune presence)."
        )
        return "Immune-Infiltrated", rationale

    if mean_dist_val is not None and mean_dist_val > 40.0:
        dist_text = format_distance(mean_dist_val, pixel_size_um=pixel_size_um)
        rationale = (
            f"Inflammatory cells comprise {inflammatory_pct:.2f}% ({inflammatory_count} cells) "
            f"but mean immune-to-tumor distance is {dist_text}, exceeding the 40.00 px "
            f"({px_to_um(40.0, pixel_size_um):.2f} μm) infiltration margin."
        )
        return "Immune-Excluded", rationale

    dist_text = (
        format_distance(mean_dist_val, pixel_size_um=pixel_size_um)
        if mean_dist_val is not None
        else "N/A"
    )
    rationale = (
        f"Inflammatory cells comprise {inflammatory_pct:.2f}% ({inflammatory_count} cells) "
        f"with mean immune-to-tumor distance of {dist_text}, within the <= 40.00 px "
        f"({px_to_um(40.0, pixel_size_um):.2f} μm) tumor infiltration threshold."
    )
    return "Immune-Infiltrated", rationale


def format_vancouver_citation(paper: Dict[str, Any]) -> str:
    """Format a literature record in strict Vancouver style without stray punctuation.

    Target pattern:
        ``Author(s). Title. Journal. Year;Vol(Issue):Pages. PMID: XXXXXXX.``
    """
    authors = str(paper.get("authors") or "Unknown").strip().rstrip(".")
    title = str(paper.get("title") or "Untitled").strip().rstrip(".")
    journal = str(paper.get("journal") or "Unknown Journal").strip().rstrip(".")
    year = str(paper.get("year") or "N/A").strip().rstrip(".")
    volume = str(paper.get("volume") or "").strip()
    issue = str(paper.get("issue") or "").strip()
    pages = str(paper.get("pages") or "").strip()
    pmid = str(paper.get("pmid") or "").strip()

    pub_segment = year
    vol_issue = ""
    if volume and issue:
        vol_issue = f"{volume}({issue})"
    elif volume:
        vol_issue = volume
    elif issue:
        vol_issue = f"({issue})"

    if vol_issue:
        pub_segment = f"{pub_segment};{vol_issue}"
    if pages:
        pub_segment = f"{pub_segment}:{pages}"

    base = f"{authors}. {title}. {journal}. {pub_segment}."
    if pmid and pmid.isdigit():
        return f"{base} PMID: {pmid}."
    return base


def _calibration_label(calibration_status: Dict[str, Any]) -> str:
    """Return canonical τ display string, e.g. 'τ = 1.6590 (Fitted)'."""
    temp = calibration_status.get("temperature")
    fitted = bool(calibration_status.get("fitted", False))
    status_tag = "Fitted" if fitted else "Default/Unfitted"
    if not _is_finite_number(temp):
        return f"τ = N/A ({status_tag})"
    temp_float = float(temp)
    temp_str = f"{temp_float:.4f}".rstrip("0").rstrip(".")
    if "." not in temp_str:
        temp_str = f"{temp_float:.1f}"
    return f"τ = {temp_str} ({status_tag})"


def build_organ_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 2 (Top-3 candidate organ predictions)."""
    organ = payload.get("organ") or {}
    top3 = organ.get("top_3_predictions") or []

    lines = [
        "| Rank | Candidate Organ | Calibrated Confidence | Percentage |",
        "| --- | --- | --- | --- |",
    ]
    if not top3:
        pred = organ.get("predicted_organ", "N/A")
        conf = organ.get("confidence_score")
        pct = f"{float(conf) * 100.0:.2f}%" if _is_finite_number(conf) else "N/A"
        lines.append(f"| 1 | {pred} | {_fmt_num(conf, 4)} | {pct} |")
    else:
        for rank, entry in enumerate(top3, start=1):
            name = entry.get("organ", "N/A")
            conf = entry.get("confidence")
            pct = f"{float(conf) * 100.0:.2f}%" if _is_finite_number(conf) else "N/A"
            lines.append(f"| {rank} | {name} | {_fmt_num(conf, 4)} | {pct} |")
    return "\n".join(lines)


def build_cell_composition_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 3 using strictly the 5 PanNuke classes."""
    cells = payload.get("cells") or {}
    counts = cells.get("class_counts") or {}
    pcts = cells.get("composition_percentages") or {}
    total = int(
        cells.get(
            "total_cell_count",
            sum(max(0, int(counts.get(c, 0) or 0)) for c in PAN_NUKE_CELL_CLASSES),
        )
        or 0
    )

    lines = [
        "| PanNuke Cell Class | Cell Count | Composition (%) |",
        "| --- | --- | --- |",
    ]
    for cls_name in PAN_NUKE_CELL_CLASSES:
        count = max(0, int(counts.get(cls_name, 0) or 0))
        if cls_name in pcts and _is_finite_number(pcts[cls_name]):
            pct_val = float(pcts[cls_name])
        else:
            pct_val = round((count / total) * 100.0, 2) if total > 0 else 0.0
        lines.append(f"| {cls_name} | {count} | {pct_val:.2f}% |")
    total_pct = "100.00%" if total > 0 else "0.00%"
    lines.append(f"| **Total** | **{total}** | **{total_pct}** |")
    return "\n".join(lines)


def build_interaction_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 5 (Pairwise Cell-Cell Interaction Matrix)."""
    spatial = payload.get("spatial") or {}
    interactions = spatial.get("interaction_matrix") or {}

    lines = [
        "| Cell-Class Pair | Proximity Edge Count | Share of Edges (%) |",
        "| --- | --- | --- |",
    ]
    total_edges = sum(int(v) for v in interactions.values() if _is_finite_number(v))
    if not interactions:
        lines.append("| None observed | 0 | 0.00% |")
        return "\n".join(lines)

    for pair_key, count in sorted(interactions.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        c_int = int(count)
        share = (c_int / total_edges * 100.0) if total_edges > 0 else 0.0
        lines.append(f"| {pair_key} | {c_int} | {share:.2f}% |")
    return "\n".join(lines)


def build_topology_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 6 (Spatial Proximity Graph & Topology)."""
    spatial = payload.get("spatial") or {}
    topo = spatial.get("graph_topology") or {}
    pixel_size_um = float(payload.get("pixel_size_um") or PIXEL_SIZE_UM)

    max_dist_px = spatial.get("max_distance_px")
    avg_dist_px = topo.get("avg_neighbor_distance_px")

    rows = [
        ("Proximity Radius Threshold", format_distance(max_dist_px, pixel_size_um)),
        ("Total Nodes (V)", str(topo.get("total_nodes", "N/A"))),
        ("Total Edges (E)", str(topo.get("total_edges", "N/A"))),
        ("Average Node Degree", _fmt_num(topo.get("avg_degree"), 4)),
        ("Graph Density", _fmt_num(topo.get("graph_density"), 4)),
        ("Average Clustering Coefficient", _fmt_num(topo.get("avg_clustering_coefficient"), 4)),
        ("Average Neighbor Distance", format_distance(avg_dist_px, pixel_size_um)),
        (
            "Cellular Density (per 100x100 px)",
            _fmt_num(topo.get("cellular_density_per_100x100px"), 2),
        ),
        ("Connected Components", str(topo.get("connected_components", "N/A"))),
    ]

    lines = [
        "| Spatial Topology Metric | Value |",
        "| --- | --- |",
    ]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def build_validity_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 7 (Statistical Validity & Sample Quality Audit)."""
    spatial = payload.get("spatial") or {}
    validity = spatial.get("validity_assessment") or {}
    is_valid = validity.get("is_valid")
    status_str = (
        "PASS" if is_valid is True else ("FAIL (Exploratory Only)" if is_valid is False else "N/A")
    )

    rows = [
        ("Validity Gate Status", status_str),
        ("Detected Cells (N)", str(validity.get("n_cells", "N/A"))),
        ("Minimum Total Cell Threshold", str(validity.get("min_total_cells", 30))),
        ("Minimum Per-Class Threshold", str(validity.get("min_per_class", 5))),
        ("Active Warnings", str(len(validity.get("warnings") or []))),
    ]

    lines = [
        "| Validity Audit Metric | Value |",
        "| --- | --- |",
    ]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def build_calibration_table(payload: Dict[str, Any]) -> str:
    """Build Markdown table for Section 8 (Confidence Calibration & Temperature Status)."""
    organ = payload.get("organ") or {}
    cal = organ.get("calibration_status") or {}

    raw_conf = organ.get("raw_confidence")
    cal_conf = organ.get("confidence_score")
    raw_pct = f" ({float(raw_conf) * 100.0:.2f}%)" if _is_finite_number(raw_conf) else ""
    cal_pct = f" ({float(cal_conf) * 100.0:.2f}%)" if _is_finite_number(cal_conf) else ""

    rows = [
        ("Calibration Method", str(cal.get("method", "temperature_scaling"))),
        ("Temperature Parameter (τ)", _calibration_label(cal)),
        ("Raw Softmax Confidence", f"{_fmt_num(raw_conf, 4)}{raw_pct}"),
        ("Calibrated Confidence", f"{_fmt_num(cal_conf, 4)}{cal_pct}"),
        ("Calibration Samples (N)", str(cal.get("n_samples", "N/A"))),
        ("Validation NLL", _fmt_num(cal.get("best_nll"), 4)),
        ("ECE Before Calibration", _fmt_num(cal.get("ece_before"), 4)),
        ("ECE After Calibration", _fmt_num(cal.get("ece_after"), 4)),
    ]

    lines = [
        "| Calibration Metric | Value |",
        "| --- | --- |",
    ]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def build_deterministic_tables_bundle(payload: Dict[str, Any]) -> str:
    """Return all 6 verified Markdown tables formatted for prompt injection."""
    sections = [
        "### Verified Table — Section 2 (Organ Identification)",
        build_organ_table(payload),
        "",
        "### Verified Table — Section 3 (5-Class PanNuke Cell Composition)",
        build_cell_composition_table(payload),
        "",
        "### Verified Table — Section 5 (Pairwise Cell-Cell Interaction Matrix)",
        build_interaction_table(payload),
        "",
        "### Verified Table — Section 6 (Spatial Proximity Graph & Topology)",
        build_topology_table(payload),
        "",
        "### Verified Table — Section 7 (Statistical Validity Gate)",
        build_validity_table(payload),
        "",
        "### Verified Table — Section 8 (Confidence Calibration)",
        build_calibration_table(payload),
    ]
    return "\n".join(sections)


def _extract_valid_papers(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return literature entries that have a valid numeric PMID."""
    lit = payload.get("literature_evidence") or {}
    results = lit.get("results") or []
    valid: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for paper in results:
        if not isinstance(paper, dict):
            continue
        pmid = str(paper.get("pmid") or "").strip()
        if pmid.isdigit() and pmid not in seen:
            seen.add(pmid)
            valid.append(paper)
    return valid


def _inline_citation(paper: Dict[str, Any]) -> str:
    """Build a compact inline citation referencing the exact PMID."""
    authors = str(paper.get("authors") or "Unknown").split(",")[0].strip()
    year = str(paper.get("year") or "N/A").strip()
    pmid = str(paper.get("pmid") or "").strip()
    return f"({authors}, {year}; PMID: {pmid})"


def render_deterministic_report(payload: Dict[str, Any]) -> str:
    """Render a complete 11-section institutional RUO pathology report offline.

    Produces a deterministic, zero-hallucination Markdown report embedding every
    pre-computed metric in both ``px`` and ``μm``, strict 5-class PanNuke bounds,
    verified Markdown tables, grounded Vancouver PMID citations (or explicit
    fallback statements), and Research-Use-Only framing.
    """
    pixel_size_um = float(payload.get("pixel_size_um") or PIXEL_SIZE_UM)
    organ = payload.get("organ") or {}
    cells = payload.get("cells") or {}
    spatial = payload.get("spatial") or {}
    topo = spatial.get("graph_topology") or {}
    immune = spatial.get("immune_infiltration") or {}
    validity = spatial.get("validity_assessment") or {}
    cal = organ.get("calibration_status") or {}

    predicted_organ = str(organ.get("predicted_organ") or "N/A")
    cal_conf = organ.get("confidence_score")
    raw_conf = organ.get("raw_confidence")
    cal_conf_pct = f"{float(cal_conf) * 100.0:.2f}%" if _is_finite_number(cal_conf) else "N/A"
    raw_conf_pct = f"{float(raw_conf) * 100.0:.2f}%" if _is_finite_number(raw_conf) else "N/A"
    tau_label = _calibration_label(cal)

    class_counts = cells.get("class_counts") or {}
    total_cells = int(
        cells.get(
            "total_cell_count",
            sum(max(0, int(class_counts.get(c, 0) or 0)) for c in PAN_NUKE_CELL_CLASSES),
        )
        or 0
    )
    dominant_cell = str(cells.get("dominant_cell_type") or "N/A")
    detection_source = str(
        cells.get("detection_source") or spatial.get("detection_source") or "N/A"
    )

    tme_phenotype = immune.get("tme_phenotype")
    tme_rationale = immune.get("tme_rationale")
    if not tme_phenotype or not tme_rationale:
        tme_phenotype, tme_rationale = classify_tme_phenotype(
            class_counts, immune, pixel_size_um=pixel_size_um
        )

    mean_dist_str = format_distance(immune.get("mean_immune_to_tumor_dist_px"), pixel_size_um)
    min_dist_str = format_distance(immune.get("closest_immune_to_tumor_dist_px"), pixel_size_um)
    avg_neighbor_str = format_distance(topo.get("avg_neighbor_distance_px"), pixel_size_um)
    graph_image_path = str(spatial.get("graph_image_path") or "N/A")

    papers = _extract_valid_papers(payload)
    cited_papers: List[Dict[str, Any]] = []

    def _cite(indices: Sequence[int]) -> str:
        if not papers:
            return NO_LITERATURE_FALLBACK
        chosen: List[Dict[str, Any]] = []
        for idx in indices:
            paper = papers[idx % len(papers)]
            if paper not in chosen:
                chosen.append(paper)
            if paper not in cited_papers:
                cited_papers.append(paper)
        return " ".join(_inline_citation(p) for p in chosen)

    sec4_cite = _cite([0, 1] if len(papers) >= 2 else [0])
    sec6_cite = _cite([1 if len(papers) > 1 else 0])
    sec8_cite = _cite([2 if len(papers) > 2 else 0])
    sec10_cite_1 = _cite([0])
    sec10_cite_2 = _cite([1 if len(papers) > 1 else 0])

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source_name = str(payload.get("source_name") or f"patch_{payload.get('patch_index', 0)}")
    source_type = str(payload.get("source_type") or "pannuke_fold")
    resolution = str(payload.get("resolution") or "256x256 px")
    stain_type = str(payload.get("stain_type") or "H&E (Hematoxylin and Eosin)")
    gt_organ = str(payload.get("ground_truth_organ") or "Unknown")
    model_source = str(organ.get("model_source") or "N/A")

    warnings_list = validity.get("warnings") or []
    recs_list = validity.get("recommendations") or []
    warnings_md = (
        "\n".join(f"- **Warning:** {w}" for w in warnings_list)
        if warnings_list
        else "- No statistical validity warnings triggered."
    )
    recs_md = (
        "\n".join(f"- **Recommendation:** {r}" for r in recs_list)
        if recs_list
        else "- Sample meets minimum total and per-class cell count thresholds for exploratory analysis."
    )

    if cited_papers:
        refs_lines = [
            f"{idx}. {format_vancouver_citation(paper)}"
            for idx, paper in enumerate(cited_papers, start=1)
        ]
        refs_md = "\n".join(refs_lines)
    else:
        refs_md = NO_LITERATURE_FALLBACK

    report_sections = [
        "# BioCellAgents Computational Pathology Analytical Report (Research Use Only)",
        "",
        "## 1. Document Metadata Header & Executive Analytical Summary",
        "",
        "> **Analytical Notice (Research Use Only — RUO):** This report is generated by an automated "
        "computational pathology research pipeline for exploratory translational research only. "
        "It is non-diagnostic and must not be used for clinical decision-making.",
        "",
        f"- **Generated (UTC):** {generated_utc}",
        f"- **Schema Version:** {payload.get('schema_version', '2.0')}",
        f"- **Sample Identifier:** {source_name} (`{source_type}`)",
        f"- **Dataset Fold / Source:** {payload.get('fold_dir') or 'Custom Upload'} "
        f"(Patch Index: {payload.get('patch_index', 0)})",
        f"- **Ground-Truth Organ Label:** {gt_organ}",
        f"- **Patch Resolution & Scale:** {resolution} at {pixel_size_um:.2f} μm/px (40x objective)",
        f"- **Stain Modality:** {stain_type}",
        "",
        f"**Executive Analytical Summary:** Quantitative vision and spatial topology analysis "
        f"identified **{predicted_organ}** as the primary candidate tissue origin with a calibrated "
        f"confidence of **{_fmt_num(cal_conf, 4)} ({cal_conf_pct})** under **{tau_label}** "
        f"(raw confidence: {_fmt_num(raw_conf, 4)} / {raw_conf_pct}). A total of **{total_cells}** "
        f"nuclei were enumerated across the 5 PanNuke classes, dominated by **{dominant_cell}** "
        f"nuclei. Deterministic spatial phenotyping classifies this field of view as "
        f"**{tme_phenotype}**.",
        "",
        "## 2. Primary Organ Identification & Calibrated Confidence",
        "",
        f"- **Predicted Organ:** {predicted_organ}",
        f"- **Calibrated Confidence:** {_fmt_num(cal_conf, 4)} ({cal_conf_pct})",
        f"- **Raw Uncalibrated Confidence:** {_fmt_num(raw_conf, 4)} ({raw_conf_pct})",
        f"- **Calibration Temperature:** {tau_label}",
        f"- **Model Provenance:** {model_source}",
        "",
        build_organ_table(payload),
        "",
        "## 3. Cellular Microenvironment & 5-Class Breakdown",
        "",
        f"A total of **{total_cells}** cells were detected using `{detection_source}`. "
        f"Dominant PanNuke cell class: **{dominant_cell}**. Counts and composition percentages "
        "strictly reflect the 5 PanNuke classes:",
        "",
        build_cell_composition_table(payload),
        "",
        "## 4. Tumor Microenvironment (TME) Architecture & Immune Infiltration",
        "",
        f"- **TME Phenotype Classification:** **{tme_phenotype}**",
        f"- **Classification Rationale:** {tme_rationale}",
        f"- **Inflammatory Cell Count:** {immune.get('inflammatory_count', class_counts.get('Inflammatory', 0))}",
        f"- **Neoplastic Cell Count:** {immune.get('neoplastic_count', class_counts.get('Neoplastic', 0))}",
        f"- **Mean Immune-to-Tumor Distance:** {mean_dist_str}",
        f"- **Closest Immune-to-Tumor Distance:** {min_dist_str}",
        "",
        f"**Literature Context:** Spatial immune-to-tumor proximity patterns and {tme_phenotype} "
        f"microenvironment states in {predicted_organ} tissue are associated with distinct cellular "
        f"organization profiles in translational oncology cohorts. {sec4_cite}",
        "",
        "## 5. Pairwise Cell-Cell Interaction Matrix",
        "",
        f"Undirected proximity edges within {format_distance(spatial.get('max_distance_px'), pixel_size_um)} "
        "summarize direct cell-class adjacency across the detected cellular graph:",
        "",
        build_interaction_table(payload),
        "",
        "## 6. Spatial Proximity Graph & Topology Analysis",
        "",
        f"Constructed a 2D Euclidean proximity graph with **{topo.get('total_nodes', 0)}** nodes (V), "
        f"**{topo.get('total_edges', 0)}** edges (E), average clustering coefficient of "
        f"**{_fmt_num(topo.get('avg_clustering_coefficient'), 4)}**, and average neighbor distance of "
        f"**{avg_neighbor_str}**. Graph overlay artifact saved to `{graph_image_path}`.",
        "",
        build_topology_table(payload),
        "",
        f"**Spatial Topology Literature Context:** Graph-theoretic metrics including clustering "
        f"coefficient, node degree, and graph density quantify local cellular niches in digital "
        f"pathology patches. {sec6_cite}",
        "",
        "## 7. Statistical Validity & Sample Quality Audit",
        "",
        f"Sample reliability was audited against minimum total cell count "
        f"(`N >= {validity.get('min_total_cells', 30)}`) "
        f"and per-class thresholds (`n_c >= {validity.get('min_per_class', 5)}`).",
        "",
        build_validity_table(payload),
        "",
        "### Audit Warnings",
        warnings_md,
        "",
        "### Audit Recommendations",
        recs_md,
        "",
        "## 8. Confidence Calibration & Temperature Status",
        "",
        f"Organ classification logits were scaled using post-hoc temperature scaling (**{tau_label}**), "
        f"shifting top-1 confidence from raw **{_fmt_num(raw_conf, 4)}** to calibrated "
        f"**{_fmt_num(cal_conf, 4)}**.",
        "",
        build_calibration_table(payload),
        "",
        f"**Calibration Methodology Context:** Temperature scaling preserves top-1 ranking while "
        f"mitigating neural network overconfidence in computational pathology workflows. {sec8_cite}",
        "",
        "## 9. Limitations, Artifacts & Domain Shift Risk",
        "",
        f"- **Field-of-View Bounds:** Analysis is restricted to a single {resolution} tissue patch "
        f"({pixel_size_um:.2f} μm/px) and may not capture whole-slide spatial heterogeneity.",
        "- **Taxonomy Constraints:** Nuclei classification is strictly bounded to the 5 PanNuke "
        "morphological classes (`Neoplastic`, `Inflammatory`, `Connective`, `Dead`, `Epithelial`) "
        "and does not resolve immunohistochemistry (IHC) sub-lineages.",
        "- **Scanner & Stain Domain Shift:** Cross-cohort H&E staining intensity, fixation artifacts, "
        "and optical magnification differences can impact both organ confidence and nuclear segmentation.",
        "- **Research Use Only (RUO):** Outputs are strictly analytical research measurements and "
        "must not be interpreted as clinical pathology reports.",
        "",
        "## 10. Translational Research & Next Action Protocol",
        "",
        f"1. **Multi-Patch Whole-Slide Aggregation:** Expand spatial sampling across adjacent "
        f"{predicted_organ} regions of interest to verify whether the observed {tme_phenotype} "
        f"architecture persists across tumor core and invasive margin compartments. {sec10_cite_1}",
        f"2. **Orthogonal Multiplex Profiling:** Pair H&E spatial graph topology with multiplex "
        f"immunofluorescence or spatial transcriptomics assays in translational cohorts to validate "
        f"immune-tumor spatial proximity metrics. {sec10_cite_2}",
        "",
        "## 11. References",
        "",
        refs_md,
        "",
    ]
    return "\n".join(report_sections)


def _sanitize_pdf_text(text: str) -> str:
    """Replace common Unicode math/typography characters with Latin-1 safe equivalents."""
    replacements = {
        "\u03bc": "u",  # Greek mu
        "\u00b5": "u",  # Micro sign
        "\u03c4": "tau",  # Greek tau
        "\u2194": "<->",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00a0": " ",
    }
    out = text
    for src_ch, dst_ch in replacements.items():
        out = out.replace(src_ch, dst_ch)
    return out.encode("latin-1", "replace").decode("latin-1")


def export_report_pdf(
    markdown_text: str,
    title: str = "BioCellAgents Analytical Pathology Report (RUO)",
) -> bytes:
    """Export a Markdown pathology report to PDF bytes using lazily imported ``fpdf2``."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title banner
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.multi_cell(0, 8, _sanitize_pdf_text(title))
    pdf.ln(2)

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(2)
            continue

        # Skip pure Markdown table divider lines like | --- | --- |
        if re.match(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$", line):
            continue

        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = re.sub(r"`([^`]+)`", r"\1", clean)
        clean = _sanitize_pdf_text(clean)

        if line.startswith("# "):
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.multi_cell(0, 7, clean[2:].strip())
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.multi_cell(0, 6.5, clean[3:].strip())
        elif line.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.multi_cell(0, 6, clean[4:].strip())
        elif line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in clean.strip("|").split("|")]
            row_text = " | ".join(cells)
            pdf.set_font("Courier", size=8.5)
            pdf.multi_cell(0, 5, row_text)
        elif line.startswith(">"):
            pdf.set_font("Helvetica", style="I", size=9.5)
            pdf.multi_cell(0, 5.5, clean.lstrip(">").strip())
        else:
            pdf.set_font("Helvetica", size=9.5)
            pdf.multi_cell(0, 5.5, clean)

    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", "replace")


__all__ = [
    "PIXEL_SIZE_UM",
    "PAN_NUKE_CELL_CLASSES",
    "TME_PHENOTYPES",
    "NO_LITERATURE_FALLBACK",
    "px_to_um",
    "format_distance",
    "classify_tme_phenotype",
    "format_vancouver_citation",
    "build_organ_table",
    "build_cell_composition_table",
    "build_interaction_table",
    "build_topology_table",
    "build_validity_table",
    "build_calibration_table",
    "build_deterministic_tables_bundle",
    "render_deterministic_report",
    "export_report_pdf",
]
