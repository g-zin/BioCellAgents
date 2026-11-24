"""Offline unit tests for src.inference.calibration.TemperatureScaler."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.inference.calibration import TemperatureScaler


class _DictLogitsModel(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        return {"logits": self.linear(x)}


class TestTemperatureScaler:
    """Behavior tests for temperature scaling calibration."""

    def test_default_state_and_repr(self):
        scaler = TemperatureScaler()
        assert scaler.temperature == 2.0
        assert scaler.fitted is False
        assert scaler.n_samples == 0
        assert scaler.best_nll is None
        assert "default" in repr(scaler)

    def test_calibrate_simplex_and_temperature_effect(self):
        logits = torch.tensor([[4.0, 1.0, 0.0]])

        scaler_unit = TemperatureScaler(default_temperature=1.0)
        raw_probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(scaler_unit.calibrate(logits), raw_probs, atol=1e-6)

        scaler_warm = TemperatureScaler(default_temperature=2.0)
        warm_probs = scaler_warm.calibrate(logits)
        assert torch.allclose(warm_probs.sum(dim=-1), torch.ones(1), atol=1e-6)
        assert warm_probs.max() < raw_probs.max()

        scaler_cool = TemperatureScaler(default_temperature=0.5)
        cool_probs = scaler_cool.calibrate(logits)
        assert torch.allclose(cool_probs.sum(dim=-1), torch.ones(1), atol=1e-6)
        assert cool_probs.max() > raw_probs.max()

        batched = torch.randn(8, 5)
        out_2d = scaler_warm.calibrate(batched)
        assert out_2d.shape == (8, 5)
        assert torch.allclose(out_2d.sum(dim=-1), torch.ones(8), atol=1e-5)

        out_1d = scaler_warm.calibrate(batched[0])
        assert out_1d.shape == (5,)
        assert torch.isclose(out_1d.sum(), torch.tensor(1.0), atol=1e-5)

    def test_calibrate_probabilities_handles_zeros_and_identity(self):
        scaler_warm = TemperatureScaler(default_temperature=2.0)
        probs_zero = torch.tensor([[1.0, 0.0, 0.0]])
        cal_zero = scaler_warm.calibrate_probabilities(probs_zero)
        assert not torch.isnan(cal_zero).any()
        assert not torch.isinf(cal_zero).any()
        assert torch.isclose(cal_zero.sum(), torch.tensor(1.0), atol=1e-5)

        scaler_unit = TemperatureScaler(default_temperature=1.0)
        probs_pos = torch.tensor([[0.7, 0.2, 0.1]])
        cal_unit = scaler_unit.calibrate_probabilities(probs_pos)
        assert torch.allclose(cal_unit, probs_pos, atol=1e-5)

    def test_fit_reduces_nll_on_overconfident_logits(self):
        torch.manual_seed(42)
        n_samples, n_classes = 200, 4
        labels = torch.randint(0, n_classes, (n_samples,))
        base_logits = torch.randn(n_samples, n_classes) * 0.5
        # Make ~75% match true class and scale by 5x to create overconfidence
        match_mask = torch.rand(n_samples) < 0.75
        for i in range(n_samples):
            target = int(labels[i].item())
            if match_mask[i]:
                base_logits[i, target] += 2.0
            else:
                wrong = (target + 1) % n_classes
                base_logits[i, wrong] += 2.0
        logits = base_logits * 5.0

        raw_nll = F.cross_entropy(logits, labels).item()
        scaler = TemperatureScaler()
        returned = scaler.fit(logits, labels)

        assert returned is scaler
        assert scaler.fitted is True
        assert scaler.n_samples == n_samples
        assert scaler.temperature > 1.0
        assert scaler.best_nll is not None
        assert scaler.best_nll <= raw_nll

    def test_fit_from_dataloader_tensor_and_dict_models_and_empty_error(self):
        torch.manual_seed(7)
        x = torch.randn(24, 4)
        y = torch.randint(0, 3, (24,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        tensor_model = nn.Linear(4, 3)
        scaler1 = TemperatureScaler().fit_from_dataloader(
            loader, tensor_model, torch.device("cpu"), max_batches=2
        )
        assert scaler1.fitted is True
        assert scaler1.n_samples == 16

        dict_model = _DictLogitsModel(4, 3)
        scaler2 = TemperatureScaler().fit_from_dataloader(
            loader, dict_model, torch.device("cpu"), max_batches=10
        )
        assert scaler2.fitted is True
        assert scaler2.n_samples == 24

        empty_loader = DataLoader(TensorDataset(x[:0], y[:0]), batch_size=8)
        with pytest.raises(ValueError, match="No batches processed"):
            TemperatureScaler().fit_from_dataloader(empty_loader, tensor_model, torch.device("cpu"))

    def test_save_and_load_roundtrip_and_missing_file(self, tmp_path):
        scaler = TemperatureScaler(default_temperature=2.0)
        scaler.temperature = 1.65
        scaler.fitted = True
        scaler.n_samples = 120
        scaler.best_nll = 0.42

        target = tmp_path / "nested" / "calibration.json"
        scaler.save(target)
        assert target.exists()

        loaded = TemperatureScaler.load(target)
        assert loaded.temperature == pytest.approx(1.65)
        assert loaded.fitted is True
        assert loaded.n_samples == 120
        assert loaded.best_nll == pytest.approx(0.42)
        assert loaded.default_temperature == pytest.approx(2.0)
        assert "fitted" in repr(loaded)

        missing = TemperatureScaler.load(tmp_path / "missing.json")
        assert missing.temperature == 2.0
        assert missing.fitted is False

    def test_invalid_temperature_and_empty_fit_raise_value_error(self):
        with pytest.raises(ValueError, match="positive finite"):
            TemperatureScaler(default_temperature=0.0)
        with pytest.raises(ValueError, match="positive finite"):
            TemperatureScaler(default_temperature=-1.5)

        scaler = TemperatureScaler()
        scaler.temperature = 0.0
        with pytest.raises(ValueError, match="positive finite"):
            scaler.calibrate(torch.tensor([[1.0, 2.0]]))
        with pytest.raises(ValueError, match="positive finite"):
            scaler.calibrate_probabilities(torch.tensor([[0.6, 0.4]]))

        with pytest.raises(ValueError, match="empty"):
            TemperatureScaler().fit(torch.empty(0, 5), torch.empty(0, dtype=torch.long))

    @pytest.mark.parametrize(
        "payload",
        [
            "{not valid json",
            "",
            "[1, 2, 3]",
            '{"temperature": null, "fitted": true}',
            '{"temperature": 0.0, "fitted": true}',
            '{"temperature": -2.0, "fitted": true}',
            '{"temperature": "abc", "fitted": true}',
        ],
    )
    def test_load_malformed_or_invalid_json_falls_back_cleanly(self, tmp_path, payload):
        bad_file = tmp_path / "bad_cal.json"
        bad_file.write_text(payload, encoding="utf-8")

        loaded = TemperatureScaler.load(bad_file)
        assert loaded.temperature == pytest.approx(2.0)
        assert loaded.fitted is False
        assert "default" in repr(loaded)
        probs = loaded.calibrate(torch.tensor([[2.0, 1.0]]))
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_compute_ece_from_logits_empty_and_validation(self):
        from src.inference.calibration import compute_ece

        logits = torch.tensor([[3.0, 1.0], [0.5, 2.5], [2.0, 0.0]])
        labels = torch.tensor([0, 1, 0])
        probs = torch.softmax(logits, dim=-1)

        ece_from_logits = compute_ece(logits, labels, n_bins=10, from_logits=True)
        ece_from_probs = compute_ece(probs, labels, n_bins=10, from_logits=False)
        assert ece_from_logits == pytest.approx(ece_from_probs, abs=1e-6)

        assert compute_ece(torch.empty(0, 3), torch.empty(0, dtype=torch.long)) == 0.0

        with pytest.raises(ValueError, match="n_bins"):
            compute_ece(probs, labels, n_bins=0)
        with pytest.raises(ValueError, match="sample count mismatch"):
            compute_ece(probs, labels[:2])

    def test_compute_ece_rejects_invalid_values_and_clamps_boundary_overshoot(self):
        from src.inference.calibration import compute_ece

        # Raw logits / > 1.0 probabilities rejected when from_logits=False
        with pytest.raises(ValueError, match="Probabilities must be in"):
            compute_ece(torch.tensor([[5.0, 1.0]]), torch.tensor([1]), from_logits=False)

        # Negative probabilities rejected when from_logits=False
        with pytest.raises(ValueError, match="Probabilities must be in"):
            compute_ece(torch.tensor([[-0.2, 0.8]]), torch.tensor([1]), from_logits=False)

        # NaN and Inf rejected
        with pytest.raises(ValueError, match="non-finite"):
            compute_ece(torch.tensor([[float("nan"), 0.5]]), torch.tensor([1]), from_logits=False)
        with pytest.raises(ValueError, match="non-finite"):
            compute_ece(torch.tensor([[float("inf"), 0.5]]), torch.tensor([1]), from_logits=True)

        # Zero-class (N, 0) tensor rejected
        with pytest.raises(ValueError, match="at least 1 class"):
            compute_ece(torch.empty(3, 0), torch.tensor([0, 0, 0]), from_logits=False)

        # Tiny float32 rounding overshoot (1.0 + 1e-7) is clamped cleanly, NOT dropped as 0.0
        overshoot = torch.tensor([[1.0 + 1e-7, 0.0]], dtype=torch.float32)
        ece_overshoot = compute_ece(overshoot, torch.tensor([1]), n_bins=15, from_logits=False)
        assert ece_overshoot == pytest.approx(1.0, abs=1e-6)

        undershoot = torch.tensor([[-1e-7, 1.0]], dtype=torch.float32)
        ece_undershoot = compute_ece(undershoot, torch.tensor([1]), n_bins=15, from_logits=False)
        assert ece_undershoot == pytest.approx(0.0, abs=1e-6)
