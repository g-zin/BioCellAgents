"""Offline unit tests for src.inference.validity_gate.ValidityGate."""

import pytest

from src.inference.validity_gate import ValidityGate


class TestValidityGate:
    """Behavior tests for minimum-sample statistical validity gating."""

    def test_check_cell_count_at_exact_boundary(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.check_cell_count(30)
        assert result["is_valid"] is True
        assert result["warning"] is None
        assert result["recommendation"] is None

    @pytest.mark.parametrize("n_cells", [29, 1, 0, -1])
    def test_check_cell_count_below_boundary(self, n_cells):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.check_cell_count(n_cells)
        assert result["is_valid"] is False
        assert str(n_cells) in result["warning"]
        assert "30" in result["warning"]
        assert result["recommendation"]

    def test_check_class_distribution_all_valid(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.check_class_distribution({"Neoplastic": 5, "Inflammatory": 12})
        assert result["is_valid"] is True
        assert result["warnings"] == []
        assert result["recommendations"] == []

    def test_check_class_distribution_failing_classes(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.check_class_distribution({"Neoplastic": 10, "Dead": 4, "Epithelial": 0})
        assert result["is_valid"] is False
        assert len(result["warnings"]) == 2
        joined = " ".join(result["warnings"])
        assert "Dead" in joined
        assert "Epithelial" in joined
        assert "Neoplastic" not in joined

    def test_check_class_distribution_empty_dict(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.check_class_distribution({})
        assert result["is_valid"] is True
        assert result["warnings"] == []

    def test_full_assessment_valid_total_but_invalid_class(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        result = gate.full_assessment(n_cells=40, counts_by_class={"Neoplastic": 38, "Dead": 2})
        assert result["is_valid"] is False
        assert len(result["warnings"]) == 1
        assert result["n_cells"] == 40
        assert result["min_total_cells"] == 30
        assert result["min_per_class"] == 5

    def test_full_assessment_none_counts_and_custom_thresholds(self):
        gate = ValidityGate(min_total_cells=10, min_per_class=3)
        result = gate.full_assessment(n_cells=10, counts_by_class=None)
        assert result["is_valid"] is True
        assert result["class_distribution_check"] is None

    def test_to_prompt_context_pass_and_fail(self):
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        pass_ctx = gate.to_prompt_context(35, {"Neoplastic": 20, "Inflammatory": 15})
        assert "Status: PASS" in pass_ctx
        assert "WARNING:" not in pass_ctx

        fail_ctx = gate.to_prompt_context(10, {"Neoplastic": 8, "Dead": 2})
        assert "Status: FAIL" in fail_ctx
        assert "WARNING:" in fail_ctx
        assert "RECOMMENDATION:" in fail_ctx

        class_only_ctx = gate.to_prompt_context(100, {"Neoplastic": 98, "Dead": 2})
        assert "Status: FAIL - per-class cell count below minimum threshold" in class_only_ctx
        assert "cell count below minimum threshold" not in class_only_ctx.replace(
            "per-class cell count below minimum threshold", ""
        )

        total_only_ctx = gate.to_prompt_context(10, {"Neoplastic": 5, "Inflammatory": 5})
        assert "Status: FAIL - cell count below minimum threshold" in total_only_ctx
