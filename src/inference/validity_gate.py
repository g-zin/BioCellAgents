"""
BioCellAgents - Statistical Validity Gate
==========================================
Provides minimum-sample-size checks to prevent statistically unreliable
conclusions from small cell counts.

Usage:
    from src.inference.validity_gate import ValidityGate

    gate = ValidityGate(min_total_cells=30, min_per_class=5)
    result = gate.full_assessment(n_cells=10, counts_by_class={"Neoplastic": 4, ...})
    # result["is_valid"] -> False
    # result["warnings"] -> ["Cell count (10) is below ..."]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidityGate:
    """
    Statistical validity gate for cell-count-based analyses.

    Parameters
    ----------
    min_total_cells : int
        Minimum total cell count required for statistically reliable
        spatial / compositional analysis. Below this threshold, results
        should be flagged as exploratory.
    min_per_class : int
        Minimum cell count per class required for that class's
        proportion to be meaningful.
    """

    min_total_cells: int = 30
    min_per_class: int = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_cell_count(self, n_cells: int) -> dict[str, Any]:
        """
        Check whether the total cell count meets the minimum threshold.

        Returns
        -------
        dict with keys: is_valid, warning (str | None), recommendation (str | None)
        """
        if n_cells >= self.min_total_cells:
            return {
                "is_valid": True,
                "warning": None,
                "recommendation": None,
            }
        return {
            "is_valid": False,
            "warning": (
                f"Cell count ({n_cells}) is below the minimum threshold of "
                f"{self.min_total_cells} for statistically reliable spatial "
                f"and compositional analysis. Results should be interpreted "
                f"as exploratory only."
            ),
            "recommendation": (
                f"Acquire additional tissue sections or increase the field-of-view "
                f"to capture at least {self.min_total_cells} cells before drawing "
                f"quantitative conclusions."
            ),
        }

    def check_class_distribution(
        self, counts_by_class: dict[str, int]
    ) -> dict[str, Any]:
        """
        Check whether each cell class has a sufficient count.

        Returns
        -------
        dict with keys: is_valid, warnings (list[str]), recommendations (list[str])
        """
        warnings: list[str] = []
        recommendations: list[str] = []

        for cell_type, count in counts_by_class.items():
            if count < self.min_per_class:
                warnings.append(
                    f"Class '{cell_type}' has only {count} cell(s) "
                    f"(minimum recommended: {self.min_per_class}). "
                    f"Proportions for this class are not statistically reliable."
                )
                recommendations.append(
                    f"Interpret '{cell_type}' proportions with caution; "
                    f"consider pooling with adjacent patches or acquiring "
                    f"additional samples."
                )

        is_valid = len(warnings) == 0
        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def full_assessment(
        self,
        n_cells: int,
        counts_by_class: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        Run all validity checks and return a consolidated assessment.

        Parameters
        ----------
        n_cells : int
            Total number of detected cells.
        counts_by_class : dict[str, int] | None
            Per-class cell counts (optional).

        Returns
        -------
        dict with keys:
            is_valid (bool)
            n_cells (int)
            min_total_cells (int)
            warnings (list[str])
            recommendations (list[str])
            class_distribution_check (dict | None)
        """
        total_check = self.check_cell_count(n_cells)
        all_warnings: list[str] = []
        all_recommendations: list[str] = []

        if total_check["warning"]:
            all_warnings.append(total_check["warning"])
        if total_check["recommendation"]:
            all_recommendations.append(total_check["recommendation"])

        class_check: dict[str, Any] | None = None
        if counts_by_class is not None:
            class_check = self.check_class_distribution(counts_by_class)
            all_warnings.extend(class_check["warnings"])
            all_recommendations.extend(class_check["recommendations"])

        is_valid = total_check["is_valid"] and (
            class_check["is_valid"] if class_check else True
        )

        return {
            "is_valid": is_valid,
            "n_cells": n_cells,
            "min_total_cells": self.min_total_cells,
            "min_per_class": self.min_per_class,
            "warnings": all_warnings,
            "recommendations": all_recommendations,
            "class_distribution_check": class_check,
        }

    def to_prompt_context(
        self, n_cells: int, counts_by_class: dict[str, int] | None = None
    ) -> str:
        """
        Generate a human-readable validity context string suitable for
        inclusion in LLM tool outputs or report prompts.
        """
        assessment = self.full_assessment(n_cells, counts_by_class)
        lines: list[str] = []
        lines.append("## Statistical Validity Assessment")
        lines.append(f"- Total cells detected: {n_cells}")
        lines.append(f"- Minimum threshold: {self.min_total_cells}")
        if assessment["is_valid"]:
            lines.append("- Status: PASS - cell count meets minimum threshold")
        else:
            lines.append("- Status: FAIL - cell count below minimum threshold")
            for w in assessment["warnings"]:
                lines.append(f"  - WARNING: {w}")
            for r in assessment["recommendations"]:
                lines.append(f"  - RECOMMENDATION: {r}")
        return "\n".join(lines)

