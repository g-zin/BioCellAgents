"""Unit tests for BioCellAgents inference metrics."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_predictions(self):
        from src.inference.metrics import compute_metrics
        predictions = [0, 1, 2, 3]
        ground_truth = [0, 1, 2, 3]
        class_names = ["A", "B", "C", "D"]
        results = compute_metrics(predictions, ground_truth, class_names)
        assert results["accuracy"] == 1.0
        assert results["macro_f1"] == 1.0
        for cls in class_names:
            assert results["per_class"][cls]["precision"] == 1.0
            assert results["per_class"][cls]["recall"] == 1.0
            assert results["per_class"][cls]["f1"] == 1.0

    def test_all_wrong_predictions(self):
        from src.inference.metrics import compute_metrics
        predictions = [1, 0, 3, 2]
        ground_truth = [0, 1, 2, 3]
        class_names = ["A", "B", "C", "D"]
        results = compute_metrics(predictions, ground_truth, class_names)
        assert results["accuracy"] == 0.0

    def test_partial_correct(self):
        from src.inference.metrics import compute_metrics
        predictions = [0, 1, 2, 0]
        ground_truth = [0, 1, 2, 3]
        class_names = ["A", "B", "C", "D"]
        results = compute_metrics(predictions, ground_truth, class_names)
        assert results["accuracy"] == 0.75

    def test_confusion_matrix_shape(self):
        from src.inference.metrics import compute_metrics
        predictions = [0, 1, 2]
        ground_truth = [0, 1, 2]
        class_names = ["A", "B", "C"]
        results = compute_metrics(predictions, ground_truth, class_names)
        cm = results["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_single_class(self):
        from src.inference.metrics import compute_metrics
        predictions = [0, 0, 0]
        ground_truth = [0, 0, 0]
        class_names = ["Only"]
        results = compute_metrics(predictions, ground_truth, class_names)
        assert results["accuracy"] == 1.0

    def test_imbalanced_classes(self):
        from src.inference.metrics import compute_metrics
        predictions = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        ground_truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        class_names = ["Neg", "Pos"]
        results = compute_metrics(predictions, ground_truth, class_names)
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["macro_f1"] <= 1.0


class TestSaveResults:
    """Tests for save_results."""

    def test_save_and_load(self):
        from src.inference.metrics import save_results
        results = {"accuracy": 0.95, "macro_f1": 0.92, "per_class": {"A": {"precision": 0.9}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "subdir" / "results.json")
            save_results(results, path)
            assert Path(path).exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["accuracy"] == 0.95

    def test_creates_parent_dirs(self):
        from src.inference.metrics import save_results
        results = {"test": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "a" / "b" / "c" / "out.json")
            save_results(results, path)
            assert Path(path).exists()