"""Pure-stdlib unit tests for src.reporting and src.evals.report_fidelity_eval."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from src.evals.report_fidelity_eval import evaluate_report_fidelity
from src.reporting import (
    NO_LITERATURE_FALLBACK,
    build_deterministic_tables_bundle,
    classify_tme_phenotype,
    format_distance,
    format_vancouver_citation,
    px_to_um,
    render_deterministic_report,
)


def _sample_payload(with_literature: bool = True) -> Dict[str, Any]:
    lit = (
        {
            "status": "success",
            "results": [
                {
                    "pmid": "35123456",
                    "authors": "Smith J, Doe A",
                    "title": "Spatial immune infiltration in breast TME",
                    "journal": "Nat Cancer",
                    "year": "2024",
                    "abstract": "Immune infiltrated tumors show higher response.",
                },
                {
                    "pmid": "36987654",
                    "authors": "Lee K, Park M",
                    "title": "Topology of tumor-immune spatial proximity graphs",
                    "journal": "Cancer Cell",
                    "year": "2023",
                    "abstract": "Clustering coefficient correlates with immune niches.",
                },
                {
                    "pmid": "37112233",
                    "authors": "Patel R, Chen L",
                    "title": "Temperature scaling calibration in digital pathology AI",
                    "journal": "NPJ Digit Med",
                    "year": "2025",
                    "abstract": "Temperature scaling lowers ECE across cohorts.",
                },
            ],
        }
        if with_literature
        else {"status": "no_results", "results": []}
    )
    return {
        "schema_version": "2.0",
        "patch_index": 7,
        "fold_dir": "data/raw/fold3",
        "source_name": "fold3_patch_7",
        "source_type": "pannuke_fold",
        "ground_truth_organ": "Breast",
        "organ": {
            "predicted_organ": "Breast",
            "confidence_score": 0.9123,
            "confidence": 0.9123,
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
                "ece": 0.0312,
            },
            "model_source": "EfficientNet-B0 (best_model.pt)",
            "ground_truth_organ": "Breast",
            "resolution": "256x256 px",
            "stain_type": "H&E",
        },
        "cells": {
            "total_cell_count": 45,
            "total_cells": 45,
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
                "recommendations": ["Aggregate across adjacent tiles for rare dead nuclei."],
            },
            "max_distance_px": 50.0,
            "graph_image_path": "reports/spatial_graph_patch_7.png",
        },
        "literature_evidence": lit,
    }


class TestReportingHelpers:
    def test_px_to_um_and_format_distance(self):
        assert px_to_um(40.0) == 10.0
        assert px_to_um(None) is None
        assert px_to_um(float("nan")) is None
        assert format_distance(40.0) == "40.00 px (10.00 μm)"
        assert format_distance(None) == "N/A"
        with pytest.raises(ValueError):
            px_to_um(10.0, pixel_size_um=0.0)

    def test_classify_tme_phenotype_all_three_states(self):
        # Immune-Desert (<5% inflammatory)
        label, _ = classify_tme_phenotype(
            {"Neoplastic": 40, "Inflammatory": 1, "Connective": 5, "Dead": 0, "Epithelial": 4},
            {"mean_immune_to_tumor_dist_px": 15.0},
        )
        assert label == "Immune-Desert"

        # Immune-Excluded (>40 px mean distance)
        label, _ = classify_tme_phenotype(
            {"Neoplastic": 20, "Inflammatory": 10, "Connective": 10, "Dead": 0, "Epithelial": 5},
            {"mean_immune_to_tumor_dist_px": 55.0},
        )
        assert label == "Immune-Excluded"

        # Immune-Infiltrated (<=40 px mean distance)
        label, _ = classify_tme_phenotype(
            {"Neoplastic": 20, "Inflammatory": 10, "Connective": 10, "Dead": 0, "Epithelial": 5},
            {"mean_immune_to_tumor_dist_px": 25.0},
        )
        assert label == "Immune-Infiltrated"

    def test_format_vancouver_citation(self):
        cit = format_vancouver_citation(
            {
                "authors": "Smith J, Doe A",
                "title": "Spatial TME.",
                "journal": "Nat Cancer",
                "year": "2024",
                "pmid": "35123456",
            }
        )
        assert "Smith J, Doe A." in cit
        assert "PMID: 35123456." in cit

    def test_deterministic_tables_bundle_contains_markdown_tables(self):
        payload = _sample_payload()
        bundle = build_deterministic_tables_bundle(payload)
        assert "| Rank | Candidate Organ |" in bundle
        assert "| Neoplastic | 20 | 44.44% |" in bundle
        assert "1.659" in bundle


class TestDeterministicReportAndFidelityGrader:
    def test_deterministic_report_scores_100_with_literature(self):
        payload = _sample_payload(with_literature=True)
        report = render_deterministic_report(payload)
        result = evaluate_report_fidelity(report, payload)
        assert result["passed"] is True
        assert result["score"] == 100.0
        assert result["violations"] == []

    def test_deterministic_report_passes_without_literature(self):
        payload = _sample_payload(with_literature=False)
        report = render_deterministic_report(payload)
        assert NO_LITERATURE_FALLBACK in report
        result = evaluate_report_fidelity(report, payload)
        assert result["passed"] is True
        assert result["score"] == 100.0

    def test_fidelity_grader_catches_hallucinated_cell_count(self):
        payload = _sample_payload()
        report = render_deterministic_report(payload)
        tampered = report.replace("| Neoplastic | 20 |", "| Neoplastic | 99 |")
        result = evaluate_report_fidelity(tampered, payload)
        assert result["passed"] is False
        assert result["score"] < 100.0
        assert any("Neoplastic" in issue for issue in result["violations"])

    def test_fidelity_grader_catches_fabricated_pmid(self):
        payload = _sample_payload()
        report = render_deterministic_report(payload)
        tampered = report + "\n4. Fake Author. Hallucinated Study. J Fake. 2025. PMID: 99999999.\n"
        result = evaluate_report_fidelity(tampered, payload)
        assert result["passed"] is False
        assert any("99999999" in issue for issue in result["violations"])

    def test_fidelity_grader_catches_prohibited_cell_class_and_diagnostic_claim(self):
        payload = _sample_payload()
        report = render_deterministic_report(payload)
        tampered = report.replace(
            "## 3. Cellular Microenvironment & 5-Class Breakdown",
            "## 3. Cellular Microenvironment & 5-Class Breakdown\n| Fibroblast | 12 | 10.00% |\n",
        )
        tampered += "\nPatient is diagnosed with invasive carcinoma; prescribe chemotherapy.\n"
        result = evaluate_report_fidelity(tampered, payload)
        assert result["passed"] is False
        assert result["checks"]["class_bounds"]["passed"] is False
        assert result["checks"]["ruo_compliance"]["passed"] is False

    def test_fidelity_grader_catches_missing_sections(self):
        payload = _sample_payload()
        minimal = "## 1. Document Metadata Header\nResearch Use Only.\n"
        result = evaluate_report_fidelity(minimal, payload)
        assert result["passed"] is False
        assert result["checks"]["structural_completeness"]["passed"] is False

    def test_fidelity_grader_accepts_extra_allowed_pmids(self):
        payload = copy.deepcopy(_sample_payload(with_literature=True))
        report = render_deterministic_report(payload)
        extended = (
            report.replace(
                "PMID: 35123456",
                "PMID: 35123456; PMID: 38000001",
                1,
            )
            + "\n4. Extra Author. Adaptive PubMed Finding. Nature Med. 2025. PMID: 38000001.\n"
        )
        fail_res = evaluate_report_fidelity(extended, payload)
        assert fail_res["checks"]["pmid_grounding"]["passed"] is False
        ok_res = evaluate_report_fidelity(extended, payload, extra_allowed_pmids=["38000001"])
        assert ok_res["checks"]["pmid_grounding"]["passed"] is True
