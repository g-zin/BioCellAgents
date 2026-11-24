"""Unit & integration tests for the BioCellAgents Dual-Layer Eval Harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.runner import main as eval_cli_main
from src.evals.runner import (
    render_eval_scorecard_markdown,
    run_full_eval_suite,
    run_report_fidelity_suite,
)


class TestLayer2ReportFidelitySuite:
    """Pure-stdlib tests for Layer 2 (Agent Report Fidelity Suite)."""

    def test_report_fidelity_suite_passes_and_catches_all_adversarial_probes(self):
        res = run_report_fidelity_suite()
        assert res["passed"] is True
        assert res["compliant_with_literature"]["score"] == 100.0
        assert res["compliant_no_literature"]["score"] == 100.0
        assert res["adversarial_catch_rate"] == 1.0
        for probe_name, probe in res["adversarial_probes"].items():
            assert probe["caught"] is True, f"Adversarial probe missed: {probe_name}"

    def test_fidelity_only_cli_writes_json_and_markdown_scorecards(self, tmp_path: Path):
        exit_code = eval_cli_main(
            [
                "--fidelity-only",
                "--output-dir",
                str(tmp_path),
                "--fail-on-error",
            ]
        )
        assert exit_code == 0
        summary_json = tmp_path / "eval_summary.json"
        scorecard_md = tmp_path / "eval_scorecard.md"
        assert summary_json.exists()
        assert scorecard_md.exists()

        decoded = json.loads(summary_json.read_text(encoding="utf-8"))
        assert decoded["passed"] is True
        assert decoded["layer2_report_fidelity"]["adversarial_catch_rate"] == 1.0
        md_text = scorecard_md.read_text(encoding="utf-8")
        assert "BioCellAgents — Dual-Layer Evaluation Scorecard" in md_text
        assert "CAUGHT" in md_text


class TestLayer1AndFullDualLayerSuite:
    """Tests requiring torch/numpy/networkx (runs in full CI and GPU environments)."""

    def test_full_synthetic_dual_layer_eval_suite(self, tmp_path: Path):
        pytest.importorskip("torch")
        pytest.importorskip("numpy")
        pytest.importorskip("networkx")

        summary = run_full_eval_suite(
            mode="synthetic",
            preferred_device="cpu",
            output_dir=tmp_path,
            seed=42,
        )
        assert summary["passed"] is True, f"Dual-layer eval failed: {summary}"
        l1 = summary["layer1_ml_vision"]
        assert l1["passed"] is True, f"Layer 1 eval failed: {l1}"
        assert l1["segmentation"]["foreground_dice"] >= 0.75
        assert l1["segmentation"]["panoptic_quality"] >= 0.70
        assert (
            l1["organ_and_calibration"]["ece_calibrated"] <= l1["organ_and_calibration"]["ece_raw"]
        )
        assert l1["spatial_and_validity"]["passed"] is True

        md = render_eval_scorecard_markdown(summary)
        assert "Layer 1 — ML, Calibration & Spatial Vision Evals" in md
        assert "Layer 2 — Agent Report Fidelity Evals" in md
        assert (tmp_path / "eval_summary.json").exists()
        assert (tmp_path / "eval_scorecard.md").exists()
