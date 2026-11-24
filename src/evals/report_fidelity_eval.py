"""Programmatic 100-point Report Fidelity Grader for BioCellAgents.

Pure Python standard library implementation shared by:
- ``ReportFidelityVerifierTool`` (CrewAI ``fidelity_auditor`` agent)
- ``scripts/run_evals.py`` (automated CI & GPU evaluation harness)
- Streamlit Dashboard Eval & Report Fidelity tab

Rubric (100 points across 6 independent checks):
1. 11-Section Structural Completeness (20 pts)
2. Verbatim Numeric Accuracy & Anti-Hallucination (25 pts)
3. Strict 5-Class PanNuke Bounds (10 pts)
4. Markdown Table Compliance & Precision <= 4 decimals (15 pts)
5. Vancouver PMID Grounding & Zero Fabricated PMIDs (20 pts)
6. Research Use Only (RUO) / Non-Diagnostic Safety Compliance (10 pts)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeGuard, Union

from src.reporting import PAN_NUKE_CELL_CLASSES, PIXEL_SIZE_UM, px_to_um

REQUIRED_SECTIONS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (
        1,
        "Document Metadata Header & Executive Analytical Summary",
        ("metadata", "executive", "summary"),
    ),
    (2, "Primary Organ Identification & Calibrated Confidence", ("organ", "identification")),
    (
        3,
        "Cellular Microenvironment & 5-Class Breakdown",
        ("cellular", "microenvironment", "5-class", "breakdown"),
    ),
    (
        4,
        "Tumor Microenvironment (TME) Architecture & Immune Infiltration",
        ("tme", "immune", "infiltration", "architecture"),
    ),
    (5, "Pairwise Cell-Cell Interaction Matrix", ("interaction", "pairwise")),
    (6, "Spatial Proximity Graph & Topology Analysis", ("spatial", "topology", "proximity graph")),
    (7, "Statistical Validity & Sample Quality Audit", ("validity", "sample quality")),
    (8, "Confidence Calibration & Temperature Status", ("calibration", "temperature")),
    (9, "Limitations, Artifacts & Domain Shift Risk", ("limitation", "artifact", "domain shift")),
    (
        10,
        "Translational Research & Next Action Protocol",
        ("translational", "next action", "protocol"),
    ),
    (11, "References", ("reference",)),
)

REQUIRED_TABLE_SECTIONS: Tuple[int, ...] = (2, 3, 5, 6, 8)

PROHIBITED_NON_PANNUKE_CLASSES: Tuple[str, ...] = (
    "Fibroblast",
    "Macrophage",
    "Endothelial",
    "T-cell",
    "B-cell",
    "Stromal",
    "Neutrophil",
    "Lymphocyte",
    "Plasma cell",
    "Dendritic",
    "Mast cell",
    "Eosinophil",
    "NK cell",
    "Natural Killer",
    "Adipocyte",
)

PROHIBITED_DIAGNOSTIC_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bdiagnosed with\b", "Contains clinical diagnosis phrasing ('diagnosed with')"),
    (r"\bprescribe\b", "Contains clinical prescription directive ('prescribe')"),
    (r"\binitiate chemotherapy\b", "Contains treatment directive ('initiate chemotherapy')"),
    (r"\bclinical diagnosis\s*:", "Contains clinical diagnosis header ('clinical diagnosis:')"),
    (r"\brecommended treatment\s*:", "Contains treatment header ('recommended treatment:')"),
    (r"\bdefinitive diagnosis\b", "Contains clinical diagnosis claim ('definitive diagnosis')"),
    (
        r"\bstart\s+(?:chemotherapy|immunotherapy|radiation|treatment)\b",
        "Contains treatment initiation directive",
    ),
)

_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_PMID_RE = re.compile(r"\bPMID\s*:\s*(\d+)\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_EXCESS_DECIMALS_RE = re.compile(r"(?<![A-Za-z0-9_])\d+\.\d{5,}(?![A-Za-z0-9_])")


def _is_finite_number(value: Any) -> TypeGuard[Union[int, float]]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _parse_sections(report_markdown: str) -> Dict[int, Dict[str, str]]:
    """Parse Markdown sections 1..11 into ``{sec_num: {'heading': ..., 'body': ...}}``."""
    headings: List[Tuple[int, str, int]] = []
    lines = report_markdown.splitlines()
    for line_idx, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        heading_text = match.group(2).strip()
        num_match = re.match(
            r"^(?:Section\s+)?(\d{1,2})\s*[\.:\-\)]\s*(.+)$", heading_text, re.IGNORECASE
        )
        sec_num: Optional[int] = None
        if num_match:
            candidate = int(num_match.group(1))
            if 1 <= candidate <= 11:
                sec_num = candidate
        else:
            lowered = heading_text.lower()
            for num, _, keywords in REQUIRED_SECTIONS:
                if any(kw in lowered for kw in keywords):
                    sec_num = num
                    break
        if sec_num is not None:
            headings.append((sec_num, heading_text, line_idx))

    sections: Dict[int, Dict[str, str]] = {}
    for pos, (sec_num, heading_text, start_line) in enumerate(headings):
        end_line = headings[pos + 1][2] if pos + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start_line + 1 : end_line]).strip()
        # Preserve first matched heading for a section number, or append if repeated
        if sec_num not in sections:
            sections[sec_num] = {"heading": heading_text, "body": body}
        else:
            sections[sec_num]["body"] = (sections[sec_num]["body"] + "\n" + body).strip()
    return sections


def _has_valid_markdown_table(text: str) -> bool:
    """Return True if ``text`` contains a Markdown table with header, separator, and data row."""
    lines = [ln.strip() for ln in text.splitlines()]
    for idx in range(1, len(lines) - 1):
        if _TABLE_SEPARATOR_RE.match(lines[idx]):
            header = lines[idx - 1]
            data = lines[idx + 1]
            if (
                header.startswith("|")
                and header.endswith("|")
                and data.startswith("|")
                and data.endswith("|")
            ):
                return True
    return False


def _extract_table_rows(text: str) -> List[List[str]]:
    """Extract all non-separator Markdown table rows as lists of stripped cell strings."""
    rows: List[List[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _number_variants(value: float, max_decimals: int = 4) -> List[str]:
    """Generate acceptable string representations for a float (1..max_decimals places)."""
    variants: Set[str] = set()
    for dec in range(1, max_decimals + 1):
        formatted = f"{value:.{dec}f}"
        variants.add(formatted)
        trimmed = formatted.rstrip("0").rstrip(".")
        if trimmed:
            variants.add(trimmed)
    if float(value).is_integer():
        variants.add(str(int(round(value))))
    return sorted(variants, key=len, reverse=True)


def _contains_number(text: str, value: float, allow_percent: bool = False) -> bool:
    """Return True if ``value`` (or optionally ``value * 100``%) appears as a token in ``text``."""
    candidates = _number_variants(value, max_decimals=4)
    if allow_percent:
        pct = value * 100.0
        for dec in (1, 2):
            pct_str = f"{pct:.{dec}f}"
            candidates.append(pct_str)
            candidates.append(f"{pct_str}%")
            trimmed = pct_str.rstrip("0").rstrip(".")
            if trimmed:
                candidates.append(f"{trimmed}%")
    for token in candidates:
        escaped = re.escape(token)
        if re.search(rf"(?<!\d){escaped}(?!\d)", text):
            return True
    return False


def _contains_integer(text: str, value: int) -> bool:
    """Return True if integer ``value`` appears as a standalone numeric token in ``text``."""
    # Accept plain integer or comma-formatted integer (e.g. 1,200)
    plain = str(value)
    comma = f"{value:,}"
    for tok in {plain, comma}:
        if re.search(rf"(?<![\d.]){re.escape(tok)}(?![\d.])", text):
            return True
    return False


def _check_structural_completeness(
    sections: Dict[int, Dict[str, str]],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    violations: List[str] = []
    fixes: List[str] = []
    valid_count = 0

    for sec_num, sec_title, _ in REQUIRED_SECTIONS:
        sec = sections.get(sec_num)
        if sec is None:
            violations.append(f"Missing required Section {sec_num}: {sec_title}.")
            fixes.append(f"Add Markdown heading '## {sec_num}. {sec_title}' with complete content.")
            continue
        body_clean = re.sub(r"\s+", "", sec["body"])
        if len(body_clean) < 20:
            violations.append(
                f"Section {sec_num} ({sec_title}) has insufficient body content (<20 non-whitespace chars)."
            )
            fixes.append(f"Expand Section {sec_num} ({sec_title}) with quantitative findings.")
            continue
        valid_count += 1

    score = round(20.0 * (valid_count / len(REQUIRED_SECTIONS)), 2)
    passed = valid_count == len(REQUIRED_SECTIONS)
    check = {
        "score": score,
        "max_score": 20.0,
        "passed": passed,
        "details": {
            "sections_found": sorted(sections.keys()),
            "valid_sections": valid_count,
            "required_sections": len(REQUIRED_SECTIONS),
        },
    }
    return check, violations, fixes


def _check_numeric_accuracy(
    report_markdown: str,
    sections: Dict[int, Dict[str, str]],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    violations: List[str] = []
    fixes: List[str] = []
    checked_items = 0
    matched_items = 0

    organ = payload.get("organ") or {}
    cells = payload.get("cells") or {}
    spatial = payload.get("spatial") or {}
    topo = spatial.get("graph_topology") or {}
    immune = spatial.get("immune_infiltration") or {}
    cal = organ.get("calibration_status") or {}
    pixel_size_um = float(payload.get("pixel_size_um") or PIXEL_SIZE_UM)

    def _record(ok: bool, violation_msg: str, fix_msg: str) -> None:
        nonlocal checked_items, matched_items
        checked_items += 1
        if ok:
            matched_items += 1
        else:
            violations.append(violation_msg)
            fixes.append(fix_msg)

    # 1. Predicted organ
    pred_organ = organ.get("predicted_organ")
    if pred_organ:
        _record(
            str(pred_organ).lower() in report_markdown.lower(),
            f"Predicted organ '{pred_organ}' missing from report.",
            f"Report exact predicted organ '{pred_organ}' in Sections 1 and 2.",
        )

    # 2. Calibrated confidence score
    conf_score = organ.get("confidence_score")
    if _is_finite_number(conf_score):
        _record(
            _contains_number(report_markdown, float(conf_score), allow_percent=True),
            f"Calibrated confidence score ({float(conf_score):.4f}) missing or altered in report.",
            f"Report exact calibrated confidence '{float(conf_score):.4f}' ({float(conf_score) * 100:.2f}%).",
        )

    # 3. Calibration temperature τ + fitted/unfitted status
    temp = cal.get("temperature")
    if _is_finite_number(temp):
        temp_ok = _contains_number(report_markdown, float(temp), allow_percent=False)
        fitted = bool(cal.get("fitted", False))
        lowered = report_markdown.lower()
        if fitted:
            status_ok = bool(re.search(r"\bfitted\b", lowered)) and not bool(
                re.search(r"\bunfitted\b", lowered)
            )
        else:
            status_ok = bool(re.search(r"\b(?:unfitted|default)\b", lowered))
        status_label = "Fitted" if fitted else "Default/Unfitted"
        _record(
            temp_ok and status_ok,
            f"Calibration temperature τ={float(temp):.4f} or fitted={fitted} status missing/incorrect.",
            f"Quote exact temperature τ = {float(temp):.4f} with status '({status_label})'.",
        )

    # 4. Total cell count
    total_cells = cells.get("total_cell_count")
    sec3_text = sections.get(3, {}).get("body", "")
    cell_search_scope = sec3_text if sec3_text else report_markdown
    if _is_finite_number(total_cells):
        total_int = int(total_cells)
        _record(
            _contains_integer(cell_search_scope, total_int),
            f"Exact total_cell_count ({total_int}) missing from Section 3.",
            f"Report exact total_cell_count = {total_int} in Section 3.",
        )

    # 5. All 5 exact PanNuke class counts & percentages + contradiction check in Section 3 table
    class_counts = cells.get("class_counts") or {}
    comp_pcts = cells.get("composition_percentages") or {}
    sec3_rows = _extract_table_rows(sec3_text)
    parsed_class_rows: Dict[str, List[str]] = {}
    for row in sec3_rows:
        if len(row) >= 2:
            label = re.sub(r"[*_`]", "", row[0]).strip().lower()
            parsed_class_rows[label] = row

    for cls_name in PAN_NUKE_CELL_CLASSES:
        if cls_name not in class_counts:
            continue
        expected_cnt = int(class_counts[cls_name])
        matched_row = parsed_class_rows.get(cls_name.lower())
        if matched_row is not None:
            # Verify the table row itself matches the exact count (prevents hallucinated edits!)
            count_cell = re.sub(r"[*_`,]", "", matched_row[1]).strip()
            row_count_ok = count_cell == str(expected_cnt)
            _record(
                row_count_ok,
                f"Section 3 table count mismatch for '{cls_name}': expected {expected_cnt}, found '{matched_row[1]}'.",
                f"Set '{cls_name}' cell count in Section 3 table to exact value {expected_cnt}.",
            )
            if (
                cls_name in comp_pcts
                and _is_finite_number(comp_pcts[cls_name])
                and len(matched_row) >= 3
            ):
                expected_pct = float(comp_pcts[cls_name])
                pct_ok = _contains_number(matched_row[2], expected_pct, allow_percent=False)
                _record(
                    pct_ok,
                    f"Section 3 percentage mismatch for '{cls_name}': expected {expected_pct:.2f}%, "
                    f"found '{matched_row[2]}'.",
                    f"Set '{cls_name}' percentage in Section 3 table to {expected_pct:.2f}%.",
                )
        else:
            # Fallback check in Section 3 prose if table row missing
            pair_re = re.compile(
                rf"\b{re.escape(cls_name)}\b[^\n]{{0,40}}(?<![\d.]){expected_cnt}(?![\d.])",
                re.IGNORECASE,
            )
            _record(
                bool(pair_re.search(cell_search_scope)),
                f"Exact count for PanNuke class '{cls_name}' ({expected_cnt}) missing in Section 3.",
                f"Include '{cls_name}' with exact count {expected_cnt} in Section 3 table.",
            )

    # Check Total row in Section 3 table if present
    if _is_finite_number(total_cells) and "total" in parsed_class_rows:
        total_row = parsed_class_rows["total"]
        total_cell = re.sub(r"[*_`,]", "", total_row[1]).strip()
        _record(
            total_cell == str(int(total_cells)),
            f"Section 3 Total row mismatch: expected {int(total_cells)}, found '{total_row[1]}'.",
            f"Set Total row in Section 3 table to exact value {int(total_cells)}.",
        )

    # 6. Graph topology: total_nodes, total_edges, avg_clustering_coefficient
    sec6_text = sections.get(6, {}).get("body", "") or report_markdown
    for key, is_int in (
        ("total_nodes", True),
        ("total_edges", True),
        ("avg_clustering_coefficient", False),
    ):
        val = topo.get(key)
        if not _is_finite_number(val):
            continue
        if is_int:
            ok = _contains_integer(sec6_text, int(val))
        else:
            ok = _contains_number(sec6_text, float(val), allow_percent=False)
        _record(
            ok,
            f"Graph topology metric '{key}' ({val}) missing or altered in Section 6.",
            f"Report exact {key} = {val} in Section 6.",
        )

    # 7. Immune infiltration distances in BOTH px and μm (0.25 μm/px consistency)
    sec4_text = sections.get(4, {}).get("body", "") or report_markdown
    for dist_key in ("mean_immune_to_tumor_dist_px", "closest_immune_to_tumor_dist_px"):
        dist_px = immune.get(dist_key)
        if not _is_finite_number(dist_px):
            continue
        px_val = float(dist_px)
        um_val = px_to_um(px_val, pixel_size_um=pixel_size_um)
        px_ok = _contains_number(sec4_text, px_val, allow_percent=False) and (
            "px" in sec4_text.lower()
        )
        um_ok = (
            um_val is not None
            and _contains_number(sec4_text, um_val, allow_percent=False)
            and ("μm" in sec4_text or "µm" in sec4_text or "um" in sec4_text.lower())
        )
        _record(
            px_ok and um_ok,
            (
                f"Immune infiltration metric '{dist_key}' must report both "
                f"{px_val:.2f} px and {um_val:.2f} μm in Section 4."
            ),
            f"Format '{dist_key}' as '{px_val:.2f} px ({um_val:.2f} μm)' in Section 4.",
        )

    score = round(25.0 * (matched_items / max(checked_items, 1)), 2)
    passed = len(violations) == 0
    check = {
        "score": score,
        "max_score": 25.0,
        "passed": passed,
        "details": {
            "checked_items": checked_items,
            "matched_items": matched_items,
        },
    }
    return check, violations, fixes


def _check_class_bounds(
    report_markdown: str,
    sections: Dict[int, Dict[str, str]],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    violations: List[str] = []
    fixes: List[str] = []

    sec3_text = sections.get(3, {}).get("body", "")
    # Also inspect any table rows whose first column looks like a cell class breakdown
    scan_targets = [sec3_text] if sec3_text else [report_markdown]
    found_prohibited: List[str] = []

    for target in scan_targets:
        for cls_name in PROHIBITED_NON_PANNUKE_CLASSES:
            pattern = re.compile(rf"\b{re.escape(cls_name)}s?\b", re.IGNORECASE)
            if pattern.search(target) and cls_name not in found_prohibited:
                found_prohibited.append(cls_name)

    for cls_name in found_prohibited:
        violations.append(
            f"Fabricated non-PanNuke cell class '{cls_name}' found in Section 3 cell breakdown."
        )
        fixes.append(
            f"Remove '{cls_name}' from Section 3; report ONLY the 5 PanNuke classes "
            f"({', '.join(PAN_NUKE_CELL_CLASSES)})."
        )

    passed = len(found_prohibited) == 0
    check = {
        "score": 10.0 if passed else 0.0,
        "max_score": 10.0,
        "passed": passed,
        "details": {"prohibited_classes_found": found_prohibited},
    }
    return check, violations, fixes


def _check_markdown_tables(
    report_markdown: str,
    sections: Dict[int, Dict[str, str]],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    violations: List[str] = []
    fixes: List[str] = []

    sections_with_tables: List[int] = []
    for sec_num in REQUIRED_TABLE_SECTIONS:
        sec_body = sections.get(sec_num, {}).get("body", "")
        if _has_valid_markdown_table(sec_body):
            sections_with_tables.append(sec_num)
        else:
            violations.append(f"Section {sec_num} is missing a valid Markdown table.")
            fixes.append(
                f"Insert a valid Markdown table (header + '| --- |' separator + rows) in Section {sec_num}."
            )

    excess_precision_cells: List[str] = []
    for row in _extract_table_rows(report_markdown):
        for cell in row:
            for match in _EXCESS_DECIMALS_RE.findall(cell):
                excess_precision_cells.append(match)

    if excess_precision_cells:
        sample = ", ".join(excess_precision_cells[:5])
        violations.append(
            f"Markdown table contains numeric values exceeding 4 decimal places: {sample}."
        )
        fixes.append("Round all numeric values in Markdown tables to at most 4 decimal places.")

    table_score = 12.0 * (len(sections_with_tables) / len(REQUIRED_TABLE_SECTIONS))
    precision_score = 3.0 if not excess_precision_cells else 0.0
    score = round(table_score + precision_score, 2)
    passed = (len(sections_with_tables) == len(REQUIRED_TABLE_SECTIONS)) and (
        not excess_precision_cells
    )
    check = {
        "score": score,
        "max_score": 15.0,
        "passed": passed,
        "details": {
            "sections_with_tables": sections_with_tables,
            "required_table_sections": list(REQUIRED_TABLE_SECTIONS),
            "excess_precision_values": excess_precision_cells,
        },
    }
    return check, violations, fixes


def _check_pmid_grounding(
    report_markdown: str,
    sections: Dict[int, Dict[str, str]],
    payload: Dict[str, Any],
    extra_allowed_pmids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """Verify 0 fabricated PMIDs, section citation floors, and Section 11 References sync."""
    violations: List[str] = []
    fixes: List[str] = []

    allowed_pmids: Set[str] = set()
    lit_results = (payload.get("literature_evidence") or {}).get("results") or []
    for entry in lit_results:
        if isinstance(entry, dict):
            pmid = str(entry.get("pmid") or "").strip()
            if pmid.isdigit():
                allowed_pmids.add(pmid)
    if extra_allowed_pmids:
        for pmid in extra_allowed_pmids:
            cleaned = str(pmid).strip()
            if cleaned.isdigit():
                allowed_pmids.add(cleaned)

    all_cited_pmids = _PMID_RE.findall(report_markdown)
    unique_cited = sorted(set(all_cited_pmids))
    fabricated_pmids = sorted(p for p in unique_cited if p not in allowed_pmids)

    if fabricated_pmids:
        violations.append(
            f"CRITICAL: Fabricated/ungrounded PMID(s) detected: {', '.join(fabricated_pmids)}."
        )
        fixes.append(
            f"Remove ungrounded PMID(s) ({', '.join(fabricated_pmids)}) and cite ONLY verified PMIDs "
            f"from precomputed_results or LiteratureSearchTool."
        )
        check = {
            "score": 0.0,
            "max_score": 20.0,
            "passed": False,
            "details": {
                "allowed_pmids": sorted(allowed_pmids),
                "cited_pmids": unique_cited,
                "fabricated_pmids": fabricated_pmids,
            },
        }
        return check, violations, fixes, fabricated_pmids

    deductions = 0.0
    fallback_phrase = "no indexed literature available"

    if allowed_pmids:
        # Check section citation floors
        sec4_body = sections.get(4, {}).get("body", "")
        sec4_pmids = set(_PMID_RE.findall(sec4_body))
        min_sec4 = min(2, len(allowed_pmids))
        if len(sec4_pmids) < min_sec4:
            deductions += 6.0
            violations.append(
                f"Section 4 cites {len(sec4_pmids)} unique PMID(s); expected at least {min_sec4}."
            )
            fixes.append(f"Cite at least {min_sec4} grounded PMID(s) in Section 4.")

        for sec_num in (6, 8, 10):
            sec_body = sections.get(sec_num, {}).get("body", "")
            sec_pmids = set(_PMID_RE.findall(sec_body))
            if not sec_pmids and fallback_phrase not in sec_body.lower():
                deductions += 3.0
                violations.append(
                    f"Section {sec_num} has no grounded PMID citation and no explicit fallback notice."
                )
                fixes.append(
                    f"Add at least 1 grounded PMID citation in Section {sec_num} "
                    "(or state 'No indexed literature available for this claim.')."
                )

        # Verify every PMID cited in Sections 1..10 appears in Section 11 References
        body_pmids: Set[str] = set()
        for sec_num in range(1, 11):
            body_pmids.update(_PMID_RE.findall(sections.get(sec_num, {}).get("body", "")))
        sec11_body = sections.get(11, {}).get("body", "")
        ref_pmids = set(_PMID_RE.findall(sec11_body))
        missing_in_refs = sorted(body_pmids - ref_pmids)
        if missing_in_refs:
            deductions += 5.0
            violations.append(
                f"PMID(s) cited in report body missing from Section 11 References: {', '.join(missing_in_refs)}."
            )
            fixes.append(
                f"Add Vancouver-formatted references to Section 11 for PMID(s): {', '.join(missing_in_refs)}."
            )
    else:
        # Literature evidence is empty -> require explicit fallback notice
        if fallback_phrase not in report_markdown.lower():
            deductions += 10.0
            violations.append(
                "Literature evidence is empty, but report omits explicit "
                "'No indexed literature available' fallback statement."
            )
            fixes.append(
                "Add 'No indexed literature available for this claim.' where literature evidence is empty."
            )

    score = round(max(0.0, 20.0 - deductions), 2)
    passed = len(violations) == 0
    check = {
        "score": score,
        "max_score": 20.0,
        "passed": passed,
        "details": {
            "allowed_pmids": sorted(allowed_pmids),
            "cited_pmids": unique_cited,
            "fabricated_pmids": [],
        },
    }
    return check, violations, fixes, fabricated_pmids


def _check_ruo_compliance(
    report_markdown: str,
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    violations: List[str] = []
    fixes: List[str] = []
    prohibited_hits: List[str] = []

    lowered = report_markdown.lower()
    has_ruo = any(
        token in lowered
        for token in ("research use only", "ruo", "non-diagnostic", "analytical research")
    )
    if not has_ruo:
        violations.append(
            "Report lacks explicit Research Use Only (RUO) / analytical non-diagnostic framing."
        )
        fixes.append(
            "Include explicit 'Research Use Only (RUO)' analytical disclaimer in Section 1."
        )

    for pattern, message in PROHIBITED_DIAGNOSTIC_PATTERNS:
        if re.search(pattern, report_markdown, re.IGNORECASE):
            prohibited_hits.append(message)
            violations.append(f"CRITICAL RUO violation: {message}.")
            fixes.append(
                "Replace clinical diagnostic or prescriptive language with analytical RUO framing."
            )

    if prohibited_hits:
        score = 0.0
    elif not has_ruo:
        score = 5.0
    else:
        score = 10.0

    passed = has_ruo and not prohibited_hits
    check = {
        "score": score,
        "max_score": 10.0,
        "passed": passed,
        "details": {
            "has_ruo_framing": has_ruo,
            "prohibited_diagnostic_hits": prohibited_hits,
        },
    }
    return check, violations, fixes, prohibited_hits


def evaluate_report_fidelity(
    report_markdown: str,
    precomputed_results: Dict[str, Any],
    extra_allowed_pmids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Grade a computational pathology Markdown report on a 100-point 6-check rubric.

    Args:
        report_markdown: Full Markdown report text to evaluate.
        precomputed_results: Canonical v2.0 ``precomputed_results`` dictionary.
        extra_allowed_pmids: Optional list of additional verified PMIDs fetched during
            agent execution (e.g. by ``literature_researcher``).

    Returns:
        Dictionary with keys ``score``, ``max_score``, ``passed``, ``checks``,
        ``violations``, and ``fix_instructions``.
    """
    text = report_markdown or ""
    payload = precomputed_results or {}
    sections = _parse_sections(text)

    c1, v1, f1 = _check_structural_completeness(sections)
    c2, v2, f2 = _check_numeric_accuracy(text, sections, payload)
    c3, v3, f3 = _check_class_bounds(text, sections)
    c4, v4, f4 = _check_markdown_tables(text, sections)
    c5, v5, f5, fabricated_pmids = _check_pmid_grounding(
        text, sections, payload, extra_allowed_pmids=extra_allowed_pmids
    )
    c6, v6, f6, ruo_prohibited = _check_ruo_compliance(text)

    checks = {
        "structural_completeness": c1,
        "numeric_accuracy": c2,
        "class_bounds": c3,
        "markdown_tables": c4,
        "pmid_grounding": c5,
        "ruo_compliance": c6,
    }
    total_score = round(sum(float(c["score"]) for c in checks.values()), 2)
    violations = v1 + v2 + v3 + v4 + v5 + v6
    fix_instructions = f1 + f2 + f3 + f4 + f5 + f6

    has_critical_failure = (
        bool(fabricated_pmids) or bool(ruo_prohibited) or (not c3["passed"]) or (not c2["passed"])
    )
    passed = (total_score >= 85.0) and (not has_critical_failure)

    return {
        "score": total_score,
        "max_score": 100.0,
        "passed": passed,
        "checks": checks,
        "violations": violations,
        "fix_instructions": fix_instructions,
    }


__all__ = [
    "REQUIRED_SECTIONS",
    "REQUIRED_TABLE_SECTIONS",
    "PROHIBITED_NON_PANNUKE_CLASSES",
    "evaluate_report_fidelity",
]
