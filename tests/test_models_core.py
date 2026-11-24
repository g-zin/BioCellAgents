"""Offline CPU unit tests for core BioCellAgents models and inference helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.inference.calibration import TemperatureScaler, compute_ece
from src.inference.evaluator import evaluate_classification as eval_inf_classification
from src.inference.evaluator import evaluate_segmentation as eval_inf_segmentation
from src.inference.metrics import compute_metrics
from src.models.cell_detector import to_uint8_rgb
from src.models.cellpose_segmenter import CellposeSegmenter, _build_cellpose_model
from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter
from src.models.evaluate import evaluate_classification as eval_model_classification
from src.models.evaluate import load_model_weights
from src.models.inference_pipeline import TwoStageCellAnalyzer, _load_weights
from src.models.organ_classifier import OrganClassifier


class _DummyMetadataTag:
    """Module-level custom class to exercise weights_only=True fallback."""

    def __init__(self, version: str):
        self.version = version


class _DummyDINOClfBackbone(nn.Module):
    """Lightweight offline DINOv2 classifier backbone stub."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)


class _DummyDINOSegBackbone(nn.Module):
    """Lightweight offline DINOv2 segmenter backbone stub."""

    def __init__(self, embed_dim: int = 384, mode: str = "patchtokens", extra_tokens: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        self.extra_tokens = extra_tokens
        self.proj = nn.Linear(3, embed_dim)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        mean_rgb = x.mean(dim=(-2, -1))  # (B, 3)
        base = self.proj(mean_rgb)  # (B, D)
        n_tokens = 256 + self.extra_tokens
        tokens = base.unsqueeze(1).expand(b, n_tokens, self.embed_dim).contiguous()
        if self.mode == "patchtokens":
            return {"x_norm_patchtokens": tokens}
        return {"x_prenorm": tokens}


class _DummyTorchSegNet(nn.Module):
    """Tiny torch segmenter predicting a centered foreground nucleus blob."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = torch.full((b, self.num_classes, h, w), -5.0, device=x.device)
        logits[:, 0, :, :] = 5.0
        logits[:, 1, h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = 10.0
        return logits


class _DummyTorchClfNet(nn.Module):
    """Tiny torch classifier returning deterministic class logits."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class _MockCellposeV4Model:
    """Stub mimicking Cellpose v4 API (channel_axis kwarg, 3-tuple return)."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def eval(self, img: np.ndarray, channel_axis: int | None = None):
        self.calls.append({"shape": img.shape, "channel_axis": channel_axis})
        h, w = img.shape[:2]
        masks = np.zeros((h, w), dtype=np.int32)
        masks[4:12, 4:12] = 1
        masks[20:28, 20:28] = 2
        return masks, [], []


class _MockCellposeLegacyModel:
    """Stub mimicking legacy Cellpose v2/v3 API (channels kwarg, 4-tuple return)."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def eval(self, img: np.ndarray, channels: List[int] | None = None):
        self.calls.append({"shape": img.shape, "channels": channels})
        h, w = img.shape[:2]
        masks = np.zeros((h, w), dtype=np.int32)
        masks[2:10, 2:10] = 3
        return masks, [], [], 30.0


class TestDINOv2Models:
    """Tests for DINOv2CellClassifier and DINOv2CellSegmenter with injected backbones."""

    def test_classifier_forward_and_classify_crops_tensor_and_numpy(self):
        backbone = _DummyDINOClfBackbone(embed_dim=384)
        model = DINOv2CellClassifier(
            model_size="vits14", num_classes=5, freeze_backbone=True, backbone=backbone
        )
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.classifier_head.parameters())

        # Empty crops early return
        assert model.classify_crops([]) == []

        # Forward with non-224 input resizes cleanly
        logits = model(torch.randn(2, 3, 64, 64))
        assert logits.shape == (2, 5)

        # Tensor crops
        tensor_crops = [torch.randn(3, 64, 64) for _ in range(3)]
        res_t = model.classify_crops(tensor_crops, batch_size=2)
        assert len(res_t) == 3
        assert [r["batch_idx"] for r in res_t] == [0, 1, 2]
        assert all(0 <= r["predicted_class"] < 5 for r in res_t)
        assert all(0.0 < r["confidence"] <= 1.0 for r in res_t)

        # NumPy crops
        np_crops = [np.random.rand(3, 64, 64).astype(np.float32) for _ in range(2)]
        res_np = model.classify_crops(np_crops, batch_size=2)
        assert len(res_np) == 2

    def test_classify_crops_moves_cpu_batch_to_model_device_bug2_regression(self, monkeypatch):
        """Regression for Bug #2: classify_crops must move crops to the model's device."""
        backbone = _DummyDINOClfBackbone(embed_dim=384)
        model = DINOv2CellClassifier(
            model_size="vits14", num_classes=5, freeze_backbone=True, backbone=backbone
        )
        # Move model parameters onto PyTorch's built-in 'meta' device (works on CPU-only CI)
        model.to("meta")

        seen_devices: List[torch.device] = []

        def _spy_forward(batch_tensor: torch.Tensor) -> torch.Tensor:
            seen_devices.append(batch_tensor.device)
            return torch.zeros(batch_tensor.shape[0], 5, dtype=torch.float32)

        monkeypatch.setattr(model, "forward", _spy_forward)

        cpu_crops = [torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        results = model.classify_crops(cpu_crops, batch_size=2)

        assert len(results) == 2
        assert seen_devices == [torch.device("meta")]

    def test_classify_crops_handles_tensors_arrays_dtypes_hwc_and_mixed_lists(self):
        backbone = _DummyDINOClfBackbone(embed_dim=384)
        model = DINOv2CellClassifier(
            model_size="vits14", num_classes=5, freeze_backbone=True, backbone=backbone
        )

        # Direct 4D batch tensor, 3D single tensor, and empty 4D tensor
        res_4d = model.classify_crops(torch.randn(2, 3, 32, 32), batch_size=4)
        assert len(res_4d) == 2
        res_3d = model.classify_crops(torch.randn(3, 32, 32), batch_size=4)
        assert len(res_3d) == 1
        assert model.classify_crops(torch.empty(0, 3, 32, 32)) == []

        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            model.classify_crops(torch.randn(3, 32))

        # [4D single-item tensor], [uint8 tensor], [float64 tensor], [HWC uint8 ndarray]
        res_4d_item = model.classify_crops([torch.randn(1, 3, 32, 32)])
        assert len(res_4d_item) == 1

        res_u8 = model.classify_crops([torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)])
        assert len(res_u8) == 1

        res_f64 = model.classify_crops([torch.randn(3, 32, 32, dtype=torch.float64)])
        assert len(res_f64) == 1

        res_hwc = model.classify_crops([np.zeros((32, 32, 3), dtype=np.uint8)])
        assert len(res_hwc) == 1

        # Mixed [float32 tensor, float64 ndarray] and reverse order
        mixed = [
            torch.randn(3, 32, 32, dtype=torch.float32),
            np.random.rand(3, 32, 32).astype(np.float64),
        ]
        res_mixed = model.classify_crops(mixed, batch_size=4)
        assert len(res_mixed) == 2
        assert [r["batch_idx"] for r in res_mixed] == [0, 1]

    def test_segmenter_patchtokens_and_prenorm_cls_tokens_defect_a7(self):
        # Standard x_norm_patchtokens (256 tokens -> 16x16)
        seg_patch = DINOv2CellSegmenter(
            model_size="vits14",
            num_classes=6,
            freeze_backbone=True,
            backbone=_DummyDINOSegBackbone(mode="patchtokens", extra_tokens=0),
        )
        out_patch = seg_patch(torch.randn(2, 3, 64, 64))
        assert out_patch.shape == (2, 6, 224, 224)

        # Fallback x_prenorm with 1 CLS token (N=257 -> strips leading CLS token)
        seg_cls = DINOv2CellSegmenter(
            model_size="vits14",
            num_classes=6,
            freeze_backbone=True,
            backbone=_DummyDINOSegBackbone(mode="prenorm", extra_tokens=1),
        )
        out_cls = seg_cls(torch.randn(2, 3, 64, 64))
        assert out_cls.shape == (2, 6, 224, 224)

        # Fallback x_prenorm with CLS + 4 register tokens (N=261)
        seg_reg = DINOv2CellSegmenter(
            model_size="vits14",
            num_classes=6,
            freeze_backbone=True,
            backbone=_DummyDINOSegBackbone(mode="prenorm", extra_tokens=5),
        )
        out_reg = seg_reg(torch.randn(1, 3, 64, 64))
        assert out_reg.shape == (1, 6, 224, 224)


class TestEvaluateClassificationAndCheckpointLoading:
    """Regression tests for Bug #6 (missing classes) and Bug #4 (safe checkpoint load)."""

    def test_evaluate_classification_missing_classes_bug6_regression(self):
        # Only classes 0 and 1 present, while target_names has 5 classes
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        images = torch.zeros(4, 3, 16, 16)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        class _ExactPairModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._idx = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                logits = torch.zeros(b, 5, device=x.device)
                for i in range(b):
                    cls = 0 if self._idx < 2 else 1
                    logits[i, cls] = 5.0
                    self._idx += 1
                return logits

        target_names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        acc, macro_f1, report = eval_model_classification(
            _ExactPairModel(), loader, torch.device("cpu"), target_names
        )
        assert acc == pytest.approx(1.0)
        assert macro_f1 == pytest.approx(2.0 / 5.0, abs=1e-4)
        for name in target_names:
            assert name in report

    def test_load_model_weights_safe_with_metadata(self, tmp_path: Path):
        model = _DummyTorchClfNet(num_classes=5)
        ckpt_path = tmp_path / "wrapped_clf.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meta": _DummyMetadataTag("v2"),
            },
            ckpt_path,
        )

        target = _DummyTorchClfNet(num_classes=5)
        with torch.no_grad():
            target.fc.weight.zero_()
        load_model_weights(target, str(ckpt_path), torch.device("cpu"))
        assert torch.allclose(target.fc.weight, model.fc.weight)


class TestCellposeSegmenterCompat:
    """Regression tests for Bug #9 (Cellpose v4 3-tuple and legacy 4-tuple APIs)."""

    def test_cellpose_v4_3tuple_and_channel_axis(self):
        mock_v4 = _MockCellposeV4Model()
        segmenter = CellposeSegmenter(device="cpu", model=mock_v4)
        img = Image.fromarray(np.zeros((32, 40, 4), dtype=np.uint8), mode="RGBA")
        mask = segmenter.segment(img)

        assert mask.shape == (32, 40)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) == 128  # two 8x8 foreground regions
        assert mock_v4.calls[0]["channel_axis"] == -1
        assert mock_v4.calls[0]["shape"] == (32, 40, 3)

    def test_cellpose_legacy_4tuple_and_channels(self):
        mock_legacy = _MockCellposeLegacyModel()
        segmenter = CellposeSegmenter(device="cpu", model=mock_legacy)
        img = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8), mode="RGB")
        mask = segmenter.segment(img)

        assert mask.shape == (24, 24)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) == 64
        assert mock_legacy.calls[0]["channels"] == [0, 0]

    def test_build_cellpose_model_var_kwargs_cuda_and_unintrospectable_value_error(
        self, monkeypatch
    ):
        import src.models.cellpose_segmenter as cp_mod

        class _VarKwargsWrapper:
            def __init__(self, **kwargs: Any):
                self.kwargs = kwargs

        cuda_dev = torch.device("cuda")
        wrapped = _build_cellpose_model(_VarKwargsWrapper, "cpsam_v2", cuda_dev)
        assert wrapped.kwargs["pretrained_model"] == "cpsam_v2"
        assert wrapped.kwargs["gpu"] is True
        assert wrapped.kwargs["device"] == cuda_dev

        class _GpuOnlyFallbackModel:
            def __init__(self, gpu: bool = False, pretrained_model: str = ""):
                self.gpu = gpu
                self.pretrained_model = pretrained_model

        def _raise_value_error(_obj: Any) -> Any:
            raise ValueError("no signature found for builtin type")

        monkeypatch.setattr(cp_mod.inspect, "signature", _raise_value_error)
        fallback_model = _build_cellpose_model(_GpuOnlyFallbackModel, "cpsam_v2", cuda_dev)
        assert fallback_model.gpu is True
        assert fallback_model.pretrained_model == "cpsam_v2"


class TestOrganClassifier:
    """Tests for OrganClassifier offline mode and no deprecation warnings (Defect A1)."""

    def test_organ_classifier_pretrained_false_no_warnings_and_features(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = OrganClassifier(num_classes=19, freeze_backbone=True, pretrained=False)

        deprecated = [w for w in caught if "pretrained" in str(w.message).lower()]
        assert deprecated == []
        assert all(not p.requires_grad for p in model.backbone.parameters())

        model.unfreeze_backbone(True)
        assert all(p.requires_grad for p in model.backbone.parameters())

        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
            logits2, features = model(x, return_features=True)

        assert logits.shape == (2, 19)
        assert logits2.shape == (2, 19)
        assert features.shape == (2, 1280)


class TestTwoStageAnalyzerAndImageHelpers:
    """Tests for RGBA handling, safe weights loading, and to_uint8_rgb (Defect A8)."""

    def test_two_stage_analyzer_rgba_pil_and_load_weights(self, tmp_path: Path):
        seg = _DummyTorchSegNet(num_classes=6)
        clf = _DummyTorchClfNet(num_classes=5)

        ckpt_path = tmp_path / "clf_wrapped.pt"
        torch.save(
            {"model_state_dict": clf.state_dict(), "tag": _DummyMetadataTag("ok")}, ckpt_path
        )
        _load_weights(clf, str(ckpt_path), torch.device("cpu"))

        analyzer = TwoStageCellAnalyzer(seg, clf, device="cpu")
        rgba_img = Image.fromarray(np.full((64, 64, 4), 180, dtype=np.uint8), mode="RGBA")
        cell_classes = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

        mask, detected = analyzer.analyze_patch(rgba_img, cell_classes, padding=4)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert len(detected) >= 1
        assert detected[0]["class_name"] in cell_classes
        assert 0.0 <= detected[0]["confidence"] <= 1.0

    def test_to_uint8_rgb_rgba_grayscale_and_float_clipping(self):
        rgba = np.full((16, 16, 4), 200, dtype=np.uint8)
        rgb_from_rgba = to_uint8_rgb(rgba)
        assert rgb_from_rgba.shape == (16, 16, 3)
        assert rgb_from_rgba.dtype == np.uint8

        gray = np.full((16, 16), 120, dtype=np.uint8)
        rgb_from_gray = to_uint8_rgb(gray)
        assert rgb_from_gray.shape == (16, 16, 3)

        float_unit = np.array([[[-0.2, 0.5, 1.0]]], dtype=np.float32)
        out_unit = to_uint8_rgb(float_unit)
        assert out_unit.dtype == np.uint8
        assert out_unit[0, 0, 0] == 0
        assert out_unit[0, 0, 1] == 127
        assert out_unit[0, 0, 2] == 255

        float_over = np.array([[[-10.0, 128.0, 300.0]]], dtype=np.float32)
        out_over = to_uint8_rgb(float_over)
        assert out_over.dtype == np.uint8
        assert out_over[0, 0].tolist() == [0, 128, 255]

    def test_to_uint8_rgb_two_channel_and_nan_inf_float_sanitization(self):
        la = np.zeros((8, 10, 2), dtype=np.uint8)
        la[:, :, 0] = 77
        la[:, :, 1] = 255
        rgb_la = to_uint8_rgb(la)
        assert rgb_la.shape == (8, 10, 3)
        assert rgb_la.dtype == np.uint8
        assert np.all(rgb_la == 77)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            all_nan = np.full((4, 4, 3), np.nan, dtype=np.float32)
            out_nan = to_uint8_rgb(all_nan)
            mixed_inf = np.array([[[np.nan, -np.inf, np.inf]]], dtype=np.float32)
            out_inf = to_uint8_rgb(mixed_inf)

        assert caught == []
        assert out_nan.shape == (4, 4, 3)
        assert out_nan.dtype == np.uint8
        assert np.all(out_nan == 0)
        assert out_inf.dtype == np.uint8
        assert out_inf[0, 0].tolist() == [0, 0, 255]


class TestCalibrationECEAndInferenceGuards:
    """Tests for compute_ece, empty dataloader guards, and missing-class macro F1."""

    def test_compute_ece_exact_and_calibration_reduction(self):
        # Perfectly calibrated: 10 samples at 0.8 confidence, 8 right, 2 wrong -> ECE = 0
        probs_perfect = torch.tensor([[0.8, 0.2]] * 10)
        labels_perfect = torch.tensor([0] * 8 + [1] * 2)
        assert compute_ece(probs_perfect, labels_perfect, n_bins=10) == pytest.approx(0.0, abs=1e-6)

        # Completely wrong at confidence 1.0 -> ECE = 1.0
        probs_wrong = torch.tensor([[1.0, 0.0]] * 4)
        labels_wrong = torch.tensor([1, 1, 1, 1])
        assert compute_ece(probs_wrong, labels_wrong, n_bins=10) == pytest.approx(1.0, abs=1e-6)

        # Overconfident logits -> TemperatureScaler reduces ECE
        torch.manual_seed(42)
        n_samples, n_classes = 200, 4
        labels = torch.randint(0, n_classes, (n_samples,))
        base = torch.randn(n_samples, n_classes) * 0.5
        correct_mask = torch.rand(n_samples) < 0.70
        for i in range(n_samples):
            t = int(labels[i].item())
            if correct_mask[i]:
                base[i, t] += 2.0
            else:
                base[i, (t + 1) % n_classes] += 2.0
        overconfident_logits = base * 6.0

        ece_before = compute_ece(overconfident_logits, labels, n_bins=15, from_logits=True)
        scaler = TemperatureScaler().fit(overconfident_logits, labels)
        cal_probs = scaler.calibrate(overconfident_logits)
        ece_after = compute_ece(cal_probs, labels, n_bins=15, from_logits=False)
        assert ece_after < ece_before

    def test_empty_dataloader_guard_and_metrics_missing_class_consistency(self):
        empty_seg_loader = DataLoader(
            TensorDataset(
                torch.empty(0, 3, 8, 8),
                torch.empty(0, 8, 8, dtype=torch.long),
            ),
            batch_size=2,
        )
        seg_res = eval_inf_segmentation(
            _DummyTorchSegNet(num_classes=4), empty_seg_loader, torch.device("cpu"), num_classes=4
        )
        assert seg_res["num_samples"] == 0
        assert seg_res["dice"] == 0.0
        assert seg_res["iou"] == 0.0
        assert seg_res["per_class_dice"] == [0.0, 0.0, 0.0, 0.0]

        empty_clf_loader = DataLoader(
            TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )
        clf_res = eval_inf_classification(
            _DummyTorchClfNet(num_classes=3), empty_clf_loader, torch.device("cpu"), ["A", "B", "C"]
        )
        assert clf_res["num_samples"] == 0
        assert clf_res["accuracy"] == 0.0

        # compute_metrics macro_f1 matches evaluator when class C has zero support
        metrics = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], ["A", "B", "C"])
        assert metrics["macro_f1"] == pytest.approx(2.0 / 3.0, abs=1e-4)
