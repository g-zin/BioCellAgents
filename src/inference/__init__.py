"""Inference and evaluation utilities for BioCellAgents."""

from src.inference.calibration import TemperatureScaler, compute_ece
from src.inference.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
    generate_report,
    run_full_evaluation,
)
from src.inference.hyperparameter_tuner import HyperparameterTuner, TuningResult
from src.inference.metrics import compute_metrics, save_results
from src.inference.validity_gate import ValidityGate

__all__ = [
    "compute_ece",
    "compute_metrics",
    "save_results",
    "evaluate_classification",
    "evaluate_segmentation",
    "run_full_evaluation",
    "generate_report",
    "HyperparameterTuner",
    "TuningResult",
    "ValidityGate",
    "TemperatureScaler",
]
