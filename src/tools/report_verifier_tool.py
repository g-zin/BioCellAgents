"""CrewAI tool wrapping the programmatic Report Fidelity Grader.

Used by the ``fidelity_auditor`` agent inside ``PathologyCrew`` to verify a
drafted 11-section computational pathology report against ground-truth
pre-computed quantitative results and retrieved PubMed/Europe PMC PMIDs before
emitting the final audited report.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.evals.report_fidelity_eval import evaluate_report_fidelity


class ReportFidelityVerifierInput(BaseModel):
    """Input arguments for the Report Fidelity Verifier Tool."""

    report_markdown: str = Field(
        ...,
        description="Full Markdown text of the drafted 11-section pathology report to audit.",
    )
    precomputed_results_json: str = Field(
        ...,
        description="Exact JSON string of the ground-truth precomputed_results payload.",
    )
    allowed_pmids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of additional verified PMIDs returned by LiteratureSearchTool "
            "during this session."
        ),
    )


class ReportFidelityVerifierTool(BaseTool):
    """Programmatically audits a pathology report against ground-truth quantitative data."""

    name: str = "Report Fidelity Verifier"
    description: str = (
        "Programmatically verifies a drafted 11-section pathology report against the "
        "ground-truth precomputed_results JSON and verified PMIDs. Checks 11-section "
        "completeness, verbatim numeric accuracy (organ, confidence, temperature tau, "
        "5-class cell counts/percentages, spatial topology, px/um distances), strict "
        "PanNuke 5-class bounds, Markdown table formatting, Vancouver PMID grounding "
        "(zero fabricated PMIDs), and Research Use Only (RUO) non-diagnostic compliance. "
        "Returns a JSON audit with pass/fail status, score (0-100), issues, and fix instructions."
    )
    args_schema: Type[BaseModel] = ReportFidelityVerifierInput

    def _run(
        self,
        report_markdown: str,
        precomputed_results_json: str,
        allowed_pmids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run programmatic fidelity evaluation and return structured audit findings."""
        try:
            payload = json.loads(precomputed_results_json)
            if not isinstance(payload, dict):
                return {
                    "passed": False,
                    "score": 0.0,
                    "error": "precomputed_results_json must decode to a JSON object.",
                }
        except (json.JSONDecodeError, TypeError) as exc:
            return {
                "passed": False,
                "score": 0.0,
                "error": f"Invalid precomputed_results_json: {exc}",
            }

        return evaluate_report_fidelity(
            report_markdown=report_markdown,
            precomputed_results=payload,
            extra_allowed_pmids=allowed_pmids,
        )
