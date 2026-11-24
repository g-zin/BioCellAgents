"""Smoke tests for CLI scripts and main.py argument parsers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_rel",
    [
        "main.py",
        "scripts/diag_dice.py",
        "scripts/verify_pipeline.py",
        "scripts/benchmark_segmenters.py",
        "scripts/fit_calibration.py",
        "scripts/evaluate_all.py",
    ],
)
def test_script_cli_help_exits_cleanly(script_rel: str):
    script_path = PROJECT_ROOT / script_rel
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{script_rel} --help failed:\n{proc.stderr}"
    assert "usage:" in proc.stdout.lower()


def test_verify_pipeline_offline_mode():
    script_path = PROJECT_ROOT / "scripts" / "verify_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--offline", "--device", "cpu"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        proc.returncode == 0
    ), f"verify_pipeline.py --offline failed:\n{proc.stderr}\n{proc.stdout}"
    assert "All BioCellAgents verification checks passed!" in proc.stdout
