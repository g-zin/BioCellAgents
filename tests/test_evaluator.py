"""Offline unit tests for src.inference.evaluator."""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.inference.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
    generate_report,
    run_full_evaluation,
)


class _ExactClassModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x encodes class index in x[:, 0]
        logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        for i in range(x.shape[0]):
            cls_idx = int(x[i, 0].item())
            logits[i, cls_idx] = 10.0
        return logits


class _FixedSegModel(nn.Module):
    def __init__(self, num_classes: int, fixed_class: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.fixed_class = fixed_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, C_or_1, H, W) where channel 0 holds target class if fixed_class is None
        b, _, h, w = x.shape
        logits = torch.full((b, self.num_classes, h, w), -10.0, device=x.device)
        if self.fixed_class is not None:
            logits[:, self.fixed_class, :, :] = 10.0
        else:
            for cls_idx in range(self.num_classes):
                mask = x[:, 0].long() == cls_idx
                logits[:, cls_idx][mask] = 10.0
        return logits


class TestEvaluator:
    """Behavior tests for classification, segmentation, dispatch, and reporting."""

    def test_evaluate_classification_perfect_and_zero_support(self):
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        features = labels.unsqueeze(1).float()
        loader = DataLoader(TensorDataset(features, labels), batch_size=2)
        model = _ExactClassModel(num_classes=3)

        res = evaluate_classification(model, loader, torch.device("cpu"), ["A", "B", "C"])
        assert res["accuracy"] == 1.0
        assert res["macro_f1"] == pytest.approx(2.0 / 3.0, abs=1e-4)
        assert res["per_class"]["C"]["support"] == 0
        assert len(res["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in res["confusion_matrix"])
        assert res["num_samples"] == 4

    def test_evaluate_segmentation_perfect_and_disjoint(self):
        masks = torch.zeros(4, 16, 16, dtype=torch.long)
        masks[:, :8, :8] = 1
        masks[:, 8:, 8:] = 2
        images = masks.unsqueeze(1).float()
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        perfect_model = _FixedSegModel(num_classes=3, fixed_class=None)
        res_perf = evaluate_segmentation(perfect_model, loader, torch.device("cpu"), num_classes=3)
        assert res_perf["dice"] == pytest.approx(1.0, abs=1e-4)
        assert res_perf["iou"] == pytest.approx(1.0, abs=1e-4)
        assert res_perf["per_class_dice"] == [1.0, 1.0, 1.0]

        zero_masks = torch.zeros(2, 16, 16, dtype=torch.long)
        zero_loader = DataLoader(
            TensorDataset(zero_masks.unsqueeze(1).float(), zero_masks), batch_size=2
        )
        disjoint_model = _FixedSegModel(num_classes=3, fixed_class=1)
        res_disj = evaluate_segmentation(
            disjoint_model, zero_loader, torch.device("cpu"), num_classes=3
        )
        assert res_disj["per_class_dice"][0] == 0.0
        assert res_disj["per_class_dice"][1] == 0.0
        assert res_disj["dice"] == 0.0

    def test_run_full_evaluation_dispatch_and_validation(self):
        labels = torch.tensor([0, 1], dtype=torch.long)
        clf_loader = DataLoader(TensorDataset(labels.unsqueeze(1).float(), labels), batch_size=2)
        clf_model = _ExactClassModel(num_classes=2)

        res_cell = run_full_evaluation(
            "cell_classification",
            clf_model,
            clf_loader,
            torch.device("cpu"),
            class_names=["A", "B"],
        )
        assert res_cell["accuracy"] == 1.0

        res_organ = run_full_evaluation(
            "organ_classification",
            clf_model,
            clf_loader,
            torch.device("cpu"),
            class_names=["O1", "O2"],
        )
        assert res_organ["accuracy"] == 1.0

        masks = torch.zeros(2, 8, 8, dtype=torch.long)
        seg_loader = DataLoader(TensorDataset(masks.unsqueeze(1).float(), masks), batch_size=2)
        seg_res = run_full_evaluation(
            "segmentation",
            _FixedSegModel(2, fixed_class=None),
            seg_loader,
            torch.device("cpu"),
            num_classes=2,
        )
        assert "dice" in seg_res

        with pytest.raises(ValueError, match="class_names required"):
            run_full_evaluation("cell_classification", clf_model, clf_loader, torch.device("cpu"))

        with pytest.raises(ValueError, match="Unknown model_type"):
            run_full_evaluation("unknown", clf_model, clf_loader, torch.device("cpu"))

    def test_generate_report_writes_json_and_formats_text(self, tmp_path):
        out_path = tmp_path / "nested" / "eval_report.json"
        results = {"accuracy": 0.95, "nested": {"f1": 0.91}, "items": list(range(10))}
        report = generate_report(results, "TestModel", output_path=str(out_path))

        assert "TestModel" in report
        assert "accuracy" in report
        assert out_path.exists()
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["accuracy"] == 0.95
