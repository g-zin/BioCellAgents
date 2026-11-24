"""Shared utilities for BioCellAgents: device, logging, metrics tracking, and helpers."""

from src.utils.logging_config import (
    InferenceLogger,
    TrainingLogger,
    get_gpu_memory_info,
    get_logger,
    setup_logger,
)
from src.utils.metrics_tracker import MetricsTracker

__all__ = [
    "setup_logger",
    "get_logger",
    "TrainingLogger",
    "InferenceLogger",
    "get_gpu_memory_info",
    "MetricsTracker",
]

try:
    from src.utils.device import (
        amp_autocast,
        create_grad_scaler,
        get_device,
        get_hardware_info,
        load_checkpoint_safe,
        pin_memory_for_device,
    )

    __all__ += [
        "get_device",
        "get_hardware_info",
        "load_checkpoint_safe",
        "amp_autocast",
        "create_grad_scaler",
        "pin_memory_for_device",
    ]
except ImportError:  # pragma: no cover - allow stdlib-only utils import without torch
    pass
