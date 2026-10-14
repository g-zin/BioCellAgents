"""Inference and evaluation utilities for BioCellAgents."""
from src.inference.metrics import compute_metrics, save_results
from src.inference.evaluator import (
    evaluate_classification,
    evaluate_segmentation,
    run_full_evaluation,
    generate_report,
)
from src.inference.hyperparameter_tuner import HyperparameterTuner, TuningResult
from src.inference.validity_gate import ValidityGate
from src.inference.calibration import TemperatureScaler

__all__ = [
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
