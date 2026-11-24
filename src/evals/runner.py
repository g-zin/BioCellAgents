"""Unified Dual-Layer Eval Harness Runner & Scorecard Generator.

Combines:
- Layer 1: ML, Calibration & Spatial Vision Evals (``src.evals.ml_vision_eval``)
- Layer 2: Agent Report Fidelity Evals (``src.evals.report_fidelity_eval``)
  including both compliant baselines and 5 adversarial hallucination probes.

Outputs ``eval_summary.json`` and ``eval_scorecard.md`` consumable by CI and
the Streamlit Dashboard Eval tab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evals.report_fidelity_eval import evaluate_report_fidelity
from src.reporting import render_deterministic_report


def _build_fidelity_benchmark_payload(with_literature: bool = True) -> Dict[str, Any]:
    lit = (
        {
            "status": "success",
            "results": [
                {
                    "pmid": "35123456",
                    "authors": "Smith J, Doe A",
                    "title": "Spatial immune infiltration in breast tumor microenvironments",
                    "journal": "Nat Cancer",
                    "year": "2024",
                    "abstract": "Immune-infiltrated tumors correlate with checkpoint response.",
                },
                {
                    "pmid": "36987654",
                    "authors": "Lee K, Park M",
                    "title": "Graph topology of cellular proximity in H&E histopathology",
                    "journal": "Cancer Cell",
                    "year": "2023",
                    "abstract": "Spatial clustering metrics stratify TME niches.",
                },
                {
                    "pmid": "37112233",
                    "authors": "Patel R, Chen L",
                    "title": "Temperature scaling calibration for computational pathology",
                    "journal": "NPJ Digit Med",
                    "year": "2025",
                    "abstract": "Post-hoc temperature scaling reduces ECE reliably.",
                },
            ],
        }
        if with_literature
        else {"status": "no_results", "results": []}
    )
    return {
        "schema_version": "2.0",
        "patch_index": 0,
        "fold_dir": "synthetic_eval",
        "source_name": "eval_benchmark_patch_0",
        "source_type": "synthetic",
        "ground_truth_organ": "Breast",
        "organ": {
            "predicted_organ": "Breast",
            "confidence_score": 0.9123,
            "raw_confidence": 0.9612,
            "top_3_predictions": [
                {"organ": "Breast", "confidence": 0.9123},
                {"organ": "Lung", "confidence": 0.0512},
                {"organ": "Colon", "confidence": 0.0211},
            ],
            "calibration_status": {
                "temperature": 1.659,
                "fitted": True,
                "method": "temperature_scaling",
                "ece": 0.0294,
            },
            "model_source": "EfficientNet-B0 (PanNuke 19-organ)",
            "ground_truth_organ": "Breast",
            "resolution": "256x256 px",
            "stain_type": "H&E",
        },
        "cells": {
            "total_cell_count": 45,
            "class_counts": {
                "Neoplastic": 20,
                "Inflammatory": 10,
                "Connective": 8,
                "Dead": 2,
                "Epithelial": 5,
            },
            "composition_percentages": {
                "Neoplastic": 44.44,
                "Inflammatory": 22.22,
                "Connective": 17.78,
                "Dead": 4.44,
                "Epithelial": 11.11,
            },
            "detection_source": "DINOv2 ViT-S/14",
        },
        "spatial": {
            "graph_topology": {
                "total_nodes": 45,
                "total_edges": 112,
                "avg_clustering_coefficient": 0.4123,
                "graph_density": 0.1131,
                "avg_neighbor_distance_px": 28.5,
                "cellular_density_per_100x100px": 6.87,
                "connected_components": 2,
            },
            "interaction_matrix": {
                "Neoplastic-Inflammatory": 24,
                "Neoplastic-Neoplastic": 40,
                "Connective-Inflammatory": 12,
            },
            "immune_infiltration": {
                "mean_immune_to_tumor_dist_px": 32.4,
                "closest_immune_to_tumor_dist_px": 11.2,
                "inflammatory_count": 10,
                "neoplastic_count": 20,
            },
            "validity_assessment": {
                "is_valid": True,
                "min_total_cells": 30,
                "min_per_class": 5,
                "warnings": ["Dead class below min_per_class (2 < 5)."],
                "recommendations": ["Pool adjacent tissue patches for rare Dead nuclei."],
            },
            "max_distance_px": 50.0,
            "graph_image_path": "reports/spatial_graph_patch_0.png",
        },
        "literature_evidence": lit,
    }


def run_report_fidelity_suite() -> Dict[str, Any]:
    """Run Layer 2 (Agent Report Fidelity) compliant + adversarial probe suite (pure stdlib)."""
    t0 = time.time()
    payload_lit = _build_fidelity_benchmark_payload(with_literature=True)
    payload_nolit = _build_fidelity_benchmark_payload(with_literature=False)

    compliant_lit = render_deterministic_report(payload_lit)
    compliant_nolit = render_deterministic_report(payload_nolit)

    res_lit = evaluate_report_fidelity(compliant_lit, payload_lit)
    res_nolit = evaluate_report_fidelity(compliant_nolit, payload_nolit)

    # 5 adversarial hallucination probes — each MUST be caught (passed == False)
    adversarial_cases = {
        "hallucinated_cell_count": compliant_lit.replace(
            "| Neoplastic | 20 |", "| Neoplastic | 99 |"
        ),
        "fabricated_pmid": compliant_lit
        + "\n4. Ghost A. Fabricated Citation. Fake J. 2025. PMID: 99999999.\n",
        "prohibited_non_pannuke_class": compliant_lit.replace(
            "## 3. Cellular Microenvironment & 5-Class Breakdown",
            "## 3. Cellular Microenvironment & 5-Class Breakdown\n| Fibroblast | 9 | 8.00% |\n",
        ),
        "diagnostic_prescription_violation": compliant_lit
        + "\nClinical diagnosis: patient is diagnosed with carcinoma; prescribe chemotherapy.\n",
        "missing_required_sections": "## 1. Document Metadata Header\nResearch Use Only.\n",
    }

    adversarial_results: Dict[str, Dict[str, Any]] = {}
    all_caught = True
    for case_name, bad_report in adversarial_cases.items():
        grade = evaluate_report_fidelity(bad_report, payload_lit)
        caught = not bool(grade["passed"])
        if not caught:
            all_caught = False
        adversarial_results[case_name] = {
            "caught": caught,
            "score": grade["score"],
            "num_violations": len(grade["violations"]),
        }

    passed = bool(res_lit["passed"] and res_nolit["passed"] and all_caught)
    return {
        "passed": passed,
        "elapsed_seconds": round(time.time() - t0, 3),
        "compliant_with_literature": {
            "passed": res_lit["passed"],
            "score": res_lit["score"],
            "violations": res_lit["violations"],
        },
        "compliant_no_literature": {
            "passed": res_nolit["passed"],
            "score": res_nolit["score"],
            "violations": res_nolit["violations"],
        },
        "adversarial_probes": adversarial_results,
        "adversarial_catch_rate": round(
            sum(1 for v in adversarial_results.values() if v["caught"])
            / max(len(adversarial_results), 1),
            4,
        ),
    }


def render_eval_scorecard_markdown(summary: Dict[str, Any]) -> str:
    """Render a clean Markdown scorecard from the dual-layer evaluation summary."""
    meta = summary["metadata"]
    layer1 = summary.get("layer1_ml_vision")
    layer2 = summary["layer2_report_fidelity"]

    lines: List[str] = [
        "# BioCellAgents — Dual-Layer Evaluation Scorecard",
        "",
        f"- **Generated (UTC):** {meta['timestamp_utc']}",
        f"- **Mode:** `{meta['mode']}`",
        f"- **Device:** `{meta['device']}`",
        f"- **Overall Gate:** **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
    ]

    if layer1 is not None:
        lines += [
            "## Layer 1 — ML, Calibration & Spatial Vision Evals",
            "",
            f"Status: **{'PASS' if layer1['passed'] else 'FAIL'}** "
            f"({layer1['elapsed_seconds']:.2f}s)",
            "",
        ]
        if layer1.get("mode") == "synthetic":
            seg = layer1["segmentation"]
            org = layer1["organ_and_calibration"]
            spa = layer1["spatial_and_validity"]
            lines += [
                "| Suite | Key Metrics | Status |",
                "|---|---|---|",
                f"| Segmentation | Fg Dice: `{seg['foreground_dice']:.4f}`, "
                f"Fg IoU: `{seg['foreground_iou']:.4f}`, PQ: `{seg['panoptic_quality']:.4f}` | "
                f"{'PASS' if seg['passed'] else 'FAIL'} |",
                f"| Organ & Calibration | Acc: `{org['accuracy']:.4f}`, Macro F1: `{org['macro_f1']:.4f}`, "
                f"ECE: `{org['ece_raw']:.4f}` -> `{org['ece_calibrated']:.4f}` "
                f"(tau=`{org['fitted_temperature']:.4f}`) | "
                f"{'PASS' if org['passed'] else 'FAIL'} |",
                f"| Spatial TME & Validity | Nodes: `{spa['infiltrated_nodes']}`, "
                f"Edges: `{spa['infiltrated_edges']}`, 3/3 TME Phenotypes, 3/3 Gate Boundaries | "
                f"{'PASS' if spa['passed'] else 'FAIL'} |",
                "",
            ]

    c_lit = layer2["compliant_with_literature"]
    c_nolit = layer2["compliant_no_literature"]
    lines += [
        "## Layer 2 — Agent Report Fidelity Evals",
        "",
        f"Status: **{'PASS' if layer2['passed'] else 'FAIL'}** "
        f"(Adversarial Catch Rate: **{layer2['adversarial_catch_rate'] * 100:.1f}%**)",
        "",
        "| Benchmark Case | Score | Result |",
        "|---|---|---|",
        f"| Compliant 11-Section Report (with PubMed PMIDs) | `{c_lit['score']:.1f}/100` | "
        f"{'PASS' if c_lit['passed'] else 'FAIL'} |",
        f"| Compliant 11-Section Report (no literature fallback) | `{c_nolit['score']:.1f}/100` | "
        f"{'PASS' if c_nolit['passed'] else 'FAIL'} |",
    ]
    for probe_name, probe in layer2["adversarial_probes"].items():
        lines.append(
            f"| Adversarial Probe: `{probe_name}` | `{probe['score']:.1f}/100` "
            f"({probe['num_violations']} violations) | "
            f"{'CAUGHT' if probe['caught'] else 'MISSED'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_full_eval_suite(
    mode: str = "synthetic",
    preferred_device: Optional[str] = None,
    output_dir: str | Path = "results/evals",
    seed: int = 42,
    max_samples: int = 0,
    fidelity_only: bool = False,
) -> Dict[str, Any]:
    """Execute both eval layers and persist JSON + Markdown scorecards."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layer1: Optional[Dict[str, Any]] = None
    device_str = "cpu"
    if not fidelity_only:
        from src.evals.ml_vision_eval import run_ml_vision_eval

        layer1 = run_ml_vision_eval(
            mode=mode,
            preferred_device=preferred_device,
            seed=seed,
            max_samples=max_samples,
            reports_dir=out_path / "artifacts",
        )
        device_str = str(layer1.get("device", "cpu"))

    layer2 = run_report_fidelity_suite()
    overall_passed = bool((layer1["passed"] if layer1 is not None else True) and layer2["passed"])

    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "passed": overall_passed,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "fidelity_only" if fidelity_only else mode,
            "device": device_str,
            "seed": seed,
        },
        "layer1_ml_vision": layer1,
        "layer2_report_fidelity": layer2,
    }

    json_file = out_path / "eval_summary.json"
    md_file = out_path / "eval_scorecard.md"
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_file.write_text(render_eval_scorecard_markdown(summary), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BioCellAgents Dual-Layer Automated Evaluation & Benchmark Harness"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="Evaluation mode: deterministic offline 'synthetic' (CI/CPU/GPU) or 'real' PanNuke fold3",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device ('auto', 'cuda', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evals",
        help="Directory for eval_summary.json and eval_scorecard.md",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--fidelity-only",
        action="store_true",
        help="Run only Layer 2 (pure-stdlib Agent Report Fidelity evals without torch/numpy)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero code if any evaluation gate fails",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preferred = None if args.device == "auto" else args.device
    summary = run_full_eval_suite(
        mode=args.mode,
        preferred_device=preferred,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples=args.max_samples,
        fidelity_only=args.fidelity_only,
    )
    print(render_eval_scorecard_markdown(summary))
    print(f"Saved JSON summary : {Path(args.output_dir) / 'eval_summary.json'}")
    print(f"Saved MD scorecard : {Path(args.output_dir) / 'eval_scorecard.md'}")

    if args.fail_on_error and not summary["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
