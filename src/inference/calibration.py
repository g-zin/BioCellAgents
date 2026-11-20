"""
BioCellAgents - Confidence Calibration (Temperature Scaling)
=============================================================
Implements temperature scaling to calibrate raw softmax confidence scores.

Temperature scaling divides logits by a learned temperature T before applying
softmax. T > 1 produces more uniform (less confident) probabilities; T < 1
produces more peaked (more confident) probabilities.

The optimal T is found by minimizing the negative log-likelihood (NLL) on a
held-out validation set.

Usage:
    from src.inference.calibration import TemperatureScaler

    # Fit on validation data
    scaler = TemperatureScaler()
    scaler.fit(logits_tensor, labels_tensor)
    print(scaler.temperature)  # e.g., 2.34

    # Calibrate new predictions
    calibrated_probs = scaler.calibrate(new_logits_tensor)

    # Save / Load
    scaler.save("checkpoints/organ/calibration.json")
    scaler = TemperatureScaler.load("checkpoints/organ/calibration.json")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler:
    """
    Temperature scaling for confidence calibration.

    Parameters
    ----------
    default_temperature : float
        Fallback temperature used when no fitted value is available.
        A value of 2.0 is conservative and prevents overconfident scores.
    """

    @staticmethod
    def _validate_positive_temperature(value: Any, name: str = "temperature") -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        return float(value)

    def __init__(self, default_temperature: float = 2.0):
        validated = self._validate_positive_temperature(
            default_temperature, name="default_temperature"
        )
        self.default_temperature: float = validated
        self.temperature: float = validated
        self.fitted: bool = False
        self.n_samples: int = 0
        self.best_nll: float | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> "TemperatureScaler":
        """
        Fit the optimal temperature by minimizing NLL on the given data.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits, shape (N, C).
        labels : torch.Tensor
            Ground-truth class indices, shape (N,).
        lr : float
            Learning rate for the temperature parameter.
        max_iter : int
            Maximum number of optimization iterations.

        Returns
        -------
        self
        """
        logits = logits.detach().float()
        labels = labels.long()

        if logits.numel() == 0 or logits.shape[0] == 0 or labels.numel() == 0:
            raise ValueError("Cannot fit TemperatureScaler on empty logits/labels.")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits and labels sample count mismatch: {logits.shape[0]} vs {labels.shape[0]}"
            )

        # Initialize T near 1.0 (log-parameterization for positivity)
        log_t = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / torch.exp(log_t)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = math.exp(log_t.item())
        self.fitted = True
        self.n_samples = logits.shape[0]
        self.best_nll = closure().item()

        return self

    def fit_from_dataloader(
        self,
        dataloader,
        model: nn.Module,
        device: torch.device,
        max_batches: int = 100,
    ) -> "TemperatureScaler":
        """
        Fit temperature by running inference on a dataloader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader yielding (images, labels) pairs.
        model : nn.Module
            Model that produces logits.
        device : torch.device
            Device for inference.
        max_batches : int
            Maximum number of batches to process.

        Returns
        -------
        self
        """
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    continue
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("output"))
                if logits is not None:
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())

        if not all_logits:
            raise ValueError("No batches processed - cannot fit temperature.")

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return self.fit(logits, labels)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to raw logits and return calibrated
        probabilities.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits, shape (..., C).

        Returns
        -------
        torch.Tensor
            Calibrated probabilities, same shape as input.
        """
        temp = self._validate_positive_temperature(self.temperature, name="temperature")
        scaled = logits / temp
        return F.softmax(scaled, dim=-1)

    def calibrate_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Approximate calibration when only probabilities (not logits) are
        available. Uses log-probabilities as a proxy for logits.

        Parameters
        ----------
        probs : torch.Tensor
            Raw softmax probabilities, shape (..., C).

        Returns
        -------
        torch.Tensor
            Calibrated probabilities, same shape as input.
        """
        temp = self._validate_positive_temperature(self.temperature, name="temperature")
        # Clip to avoid log(0)
        probs_clipped = probs.clamp(min=1e-8)
        log_probs = torch.log(probs_clipped)
        return F.softmax(log_probs / temp, dim=-1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "temperature": self.temperature,
            "fitted": self.fitted,
            "n_samples": self.n_samples,
            "best_nll": self.best_nll,
            "default_temperature": self.default_temperature,
        }

    def save(self, path: str | Path) -> None:
        """Save calibration parameters to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        """Load calibration parameters from a JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()  # Return default (unfitted) scaler
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return cls()

        if not isinstance(data, dict):
            return cls()

        raw_default = data.get("default_temperature", 2.0)
        try:
            default_temp = cls._validate_positive_temperature(
                raw_default, name="default_temperature"
            )
        except ValueError:
            default_temp = 2.0

        scaler = cls(default_temperature=default_temp)
        try:
            temp = cls._validate_positive_temperature(data.get("temperature"), name="temperature")
        except ValueError:
            return scaler

        scaler.temperature = temp
        scaler.fitted = bool(data.get("fitted", False))
        raw_n = data.get("n_samples", 0)
        scaler.n_samples = (
            int(raw_n) if isinstance(raw_n, (int, float)) and not isinstance(raw_n, bool) else 0
        )
        raw_nll = data.get("best_nll")
        scaler.best_nll = (
            float(raw_nll)
            if isinstance(raw_nll, (int, float))
            and not isinstance(raw_nll, bool)
            and math.isfinite(raw_nll)
            else None
        )
        return scaler

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "default"
        return (
            f"TemperatureScaler(T={self.temperature:.4f}, " f"status={status}, n={self.n_samples})"
        )
