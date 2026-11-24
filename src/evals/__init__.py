"""Dual-layer evaluation and report fidelity auditing package for BioCellAgents."""

from src.evals.report_fidelity_eval import evaluate_report_fidelity
from src.evals.runner import (
    render_eval_scorecard_markdown,
    run_full_eval_suite,
    run_report_fidelity_suite,
)

__all__ = [
    "evaluate_report_fidelity",
    "render_eval_scorecard_markdown",
    "run_full_eval_suite",
    "run_report_fidelity_suite",
]
