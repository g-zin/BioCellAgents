"""Offline CPU unit tests for src.utils.device."""

from __future__ import annotations

import contextlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.utils.device as device_mod
from src.utils.device import (
    amp_autocast,
    create_grad_scaler,
    get_device,
    get_hardware_info,
    load_checkpoint_safe,
    pin_memory_for_device,
)


class _LegacyCheckpointMeta:
    """Module-level custom class to trigger PyTorch 2.7+ weights_only=True fallback."""

    def __init__(self, note: str):
        self.note = note


class TestGetDevice:
    """Tests for device resolution and fallback behavior."""

    def test_auto_and_explicit_cpu_when_accelerators_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: False)

        assert get_device() == torch.device("cpu")
        assert get_device("auto") == torch.device("cpu")
        assert get_device("cpu") == torch.device("cpu")
        assert get_device(torch.device("cpu")) == torch.device("cpu")

    def test_explicit_cuda_and_mps_fall_back_to_cpu_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: False)

        assert get_device("cuda") == torch.device("cpu")
        assert get_device("cuda:0") == torch.device("cpu")
        assert get_device("mps") == torch.device("cpu")

    def test_mocked_cuda_priority_and_index_bounds(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: True)

        assert get_device() == torch.device("cuda")
        assert get_device("auto") == torch.device("cuda")
        assert get_device("cuda:0") == torch.device("cuda:0")
        assert get_device("cpu") == torch.device("cpu")
        # Out-of-range GPU index falls back safely to CPU
        assert get_device("cuda:3") == torch.device("cpu")

    def test_mocked_mps_priority_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: True)

        assert get_device() == torch.device("mps")
        assert get_device("mps") == torch.device("mps")
        assert get_device("cpu") == torch.device("cpu")


class TestGetHardwareInfo:
    """Tests for JSON-serializable hardware telemetry."""

    EXPECTED_KEYS = {
        "device",
        "device_type",
        "cuda_available",
        "mps_available",
        "device_count",
        "gpu_name",
        "cuda_version",
        "torch_version",
        "vram_total_gb",
        "vram_allocated_gb",
        "vram_reserved_gb",
        "vram_free_gb",
    }

    def test_cpu_hardware_info_schema_and_json_serializable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: False)

        info = get_hardware_info("cpu")
        assert set(info.keys()) == self.EXPECTED_KEYS
        assert info["device"] == "cpu"
        assert info["device_type"] == "cpu"
        assert info["cuda_available"] is False
        assert info["mps_available"] is False
        assert info["device_count"] == 0
        assert info["gpu_name"] is None
        assert info["vram_total_gb"] is None
        assert isinstance(info["torch_version"], str)
        assert json.loads(json.dumps(info)) == info

    def test_mocked_cuda_hardware_info(self, monkeypatch):
        gb = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "Mock NVIDIA RTX 4090")
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (20 * gb, 24 * gb))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx=0: 1 * gb)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: 2 * gb)
        monkeypatch.setattr(device_mod, "_is_mps_available", lambda: False)

        info = get_hardware_info("cuda")
        assert set(info.keys()) == self.EXPECTED_KEYS
        assert info["device"] == "cuda:0"
        assert info["device_type"] == "cuda"
        assert info["cuda_available"] is True
        assert info["device_count"] == 1
        assert info["gpu_name"] == "Mock NVIDIA RTX 4090"
        assert info["vram_total_gb"] == pytest.approx(24.0)
        assert info["vram_free_gb"] == pytest.approx(20.0)
        assert info["vram_allocated_gb"] == pytest.approx(1.0)
        assert info["vram_reserved_gb"] == pytest.approx(2.0)
        assert json.loads(json.dumps(info)) == info


class TestLoadCheckpointSafe:
    """Tests for PyTorch 2.7+ safe checkpoint loading."""

    def test_load_raw_state_dict(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        ckpt_path = tmp_path / "raw_weights.pt"
        torch.save(model.state_dict(), ckpt_path)

        loaded = load_checkpoint_safe(ckpt_path, map_location="cpu", extract_state_dict=False)
        assert "weight" in loaded and "bias" in loaded
        assert torch.allclose(loaded["weight"], model.weight)

        extracted = load_checkpoint_safe(ckpt_path, map_location="cpu", extract_state_dict=True)
        assert torch.allclose(extracted["weight"], model.weight)

    def test_load_trainer_dict_with_non_tensor_metadata_fallback(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        ckpt_path = tmp_path / "trainer_ckpt.pt"
        payload = {
            "epoch": 5,
            "best_metric": 0.91,
            "model_state_dict": model.state_dict(),
            "legacy_meta": _LegacyCheckpointMeta("v1"),
            "source_path": Path("checkpoints/demo.pt"),
        }
        torch.save(payload, ckpt_path)

        full = load_checkpoint_safe(ckpt_path, map_location="cpu", extract_state_dict=False)
        assert full["epoch"] == 5
        assert full["best_metric"] == pytest.approx(0.91)
        assert isinstance(full["legacy_meta"], _LegacyCheckpointMeta)
        assert "model_state_dict" in full

        state_dict = load_checkpoint_safe(ckpt_path, map_location="cpu", extract_state_dict=True)
        assert "weight" in state_dict and "bias" in state_dict
        assert torch.allclose(state_dict["weight"], model.weight)

    def test_missing_or_empty_path_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint_safe(tmp_path / "does_not_exist.pt")
        with pytest.raises(FileNotFoundError):
            load_checkpoint_safe("")

    def test_weights_only_unsupported_type_error_retries_without_kwarg(
        self, tmp_path: Path, monkeypatch
    ):
        ckpt_path = tmp_path / "legacy_torch.pt"
        ckpt_path.write_bytes(b"stub")

        calls: list[dict[str, object]] = []

        def _fake_load(f, map_location=None, **kwargs):
            calls.append({"map_location": map_location, "kwargs": dict(kwargs)})
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return {"model_state_dict": {"ok": 1}}

        monkeypatch.setattr(torch, "load", _fake_load)

        loaded = load_checkpoint_safe(ckpt_path, map_location="cpu", extract_state_dict=True)
        assert loaded == {"ok": 1}
        assert calls == [
            {"map_location": "cpu", "kwargs": {"weights_only": True}},
            {"map_location": "cpu", "kwargs": {}},
        ]


class TestAmpAndDataLoaderHelpers:
    """Tests for amp_autocast, create_grad_scaler, and pin_memory_for_device."""

    def test_amp_autocast_cpu_and_mocked_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        ctx = amp_autocast("cpu", enabled=True)
        assert isinstance(ctx, contextlib.nullcontext)
        with ctx:
            val = torch.tensor([1.0]) + 1.0
        assert val.item() == 2.0

        recorded: list[tuple[str, object]] = []

        def _fake_autocast(device_type: str, dtype=None):
            recorded.append((device_type, dtype))
            return contextlib.nullcontext()

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.amp, "autocast", _fake_autocast)

        with amp_autocast("cuda", enabled=True):
            pass
        with amp_autocast("CUDA", enabled=True):
            pass
        assert recorded == [("cuda", torch.float16), ("cuda", torch.float16)]

        disabled_ctx = amp_autocast("cuda", enabled=False)
        assert isinstance(disabled_ctx, contextlib.nullcontext)

    def test_create_grad_scaler_cpu_returns_none_with_zero_warnings(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scaler = create_grad_scaler("cpu", enabled=True)
        assert scaler is None
        assert caught == []

        sentinel = SimpleNamespace(tag="mock_scaler")
        captured: list[tuple[str, bool]] = []

        def _fake_grad_scaler(device: str, enabled: bool = True):
            captured.append((device, enabled))
            return sentinel

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.amp, "GradScaler", _fake_grad_scaler)

        assert create_grad_scaler("cuda", enabled=False) is None
        assert create_grad_scaler("cpu", enabled=True) is None
        assert create_grad_scaler("cuda", enabled=True) is sentinel
        assert create_grad_scaler(" cuda ", enabled=True) is sentinel
        assert captured == [("cuda", True), ("cuda", True)]

    def test_pin_memory_for_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert pin_memory_for_device("cpu") is False
        assert pin_memory_for_device("cuda") is False
        assert pin_memory_for_device("CUDA") is False

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert pin_memory_for_device("cpu") is False
        assert pin_memory_for_device("cuda") is True
        assert pin_memory_for_device("CUDA") is True
        assert pin_memory_for_device("  cuda:0  ") is True
        assert pin_memory_for_device(torch.device("cuda:0")) is True


class TestLoggingConfigLogDir:
    """Regression test for Defect A11: TrainingLogger/InferenceLogger must honor log_dir."""

    def test_training_and_inference_loggers_write_to_custom_log_dir(self, tmp_path: Path):
        from src.utils.logging_config import InferenceLogger, TrainingLogger

        train_dir = tmp_path / "train_logs"
        infer_dir = tmp_path / "infer_logs"

        tlog = TrainingLogger(task_name="phase2_seg", log_dir=str(train_dir))
        ilog = InferenceLogger(model_name="phase2_organ", log_dir=str(infer_dir))
        try:
            tlog.log_epoch_start(epoch=1, total_epochs=2)
            ilog.log_result(predicted_organ="Breast", confidence=0.92)
            assert (train_dir / "biocell.phase2_seg.log").is_file()
            assert (infer_dir / "biocell.phase2_organ.log").is_file()
        finally:
            for logger in (tlog.logger, ilog.logger):
                for handler in list(logger.handlers):
                    handler.close()
                logger.handlers.clear()
