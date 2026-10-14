"""Centralized logging configuration for BioCellAgents.

Provides structured logging with:
- Console output (human-readable, color-coded)
- File output (JSON-structured for programmatic parsing)
- Training-specific loggers with epoch/step context
- Inference-specific loggers with latency tracking
- GPU memory monitoring
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured log parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for key in ("epoch", "step", "loss", "metric", "gpu_mem_mb", "inference_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class ColorConsoleFormatter(logging.Formatter):
    """Color-coded console formatter for human-readable output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = f"{color}[{timestamp}] {record.levelname:<8} {record.name}: {record.getMessage()}{self.RESET}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def setup_logger(
    name: str = "biocell",
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """Create and configure a logger with console + rotating file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path / f"{name}.log",
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorConsoleFormatter())
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "biocell") -> logging.Logger:
    """Get an existing logger or create a default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger



class TrainingLogger:
    """Specialized logger for training loops with epoch/step context."""

    def __init__(self, task_name: str = "training", log_dir: str = "logs"):
        self.task_name = task_name
        self.logger = get_logger(f"biocell.{task_name}")
        self._epoch_start_time: Optional[float] = None

    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self._epoch_start_time = time.time()
        self.logger.info(
            f"{'='*60}\nEpoch {epoch}/{total_epochs} [{self.task_name}]\n{'='*60}",
            extra={"epoch": epoch},
        )

    def log_batch(self, step: int, loss: float, lr: float = 0.0) -> None:
        self.logger.debug(
            f"Step {step} | Loss: {loss:.6f} | LR: {lr:.2e}",
            extra={"step": step, "loss": loss},
        )

    def log_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float,
        val_metric: float, metric_name: str = "dice",
    ) -> None:
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        self.logger.info(
            f"Epoch {epoch} complete | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val {metric_name.upper()}: {val_metric:.4f} "
            f"| Time: {elapsed:.1f}s",
            extra={"epoch": epoch, "loss": train_loss, "metric": val_metric},
        )

    def log_checkpoint(self, path: str, is_best: bool = False) -> None:
        status = "BEST" if is_best else "saved"
        self.logger.info(f"Checkpoint {status}: {path}")

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        if exc:
            self.logger.error(f"{message}: {exc}", exc_info=exc)
        else:
            self.logger.error(message)


class InferenceLogger:
    """Specialized logger for inference with latency tracking."""

    def __init__(self, model_name: str = "inference", log_dir: str = "logs"):
        self.model_name = model_name
        self.logger = get_logger(f"biocell.{model_name}")

    def track(self, operation: str):
        """Context manager that tracks operation duration."""
        return _TimingContext(self.logger, operation)

    def log_result(self, **kwargs) -> None:
        self.logger.info(f"Inference result: {json.dumps(kwargs, default=str)}")


class _TimingContext:
    """Context manager for timing operations and logging duration."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self._start: float = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self._start) * 1000
        if exc_type:
            self.logger.error(
                f"{self.operation} FAILED in {elapsed_ms:.1f}ms: {exc_val}",
                extra={"inference_ms": elapsed_ms},
            )
        else:
            self.logger.info(
                f"{self.operation} completed in {elapsed_ms:.1f}ms",
                extra={"inference_ms": elapsed_ms},
            )
        return False


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage (if CUDA available)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "available_mb": round(free / 1024**2, 1),
                "used_mb": round((total - free) / 1024**2, 1),
                "total_mb": round(total / 1024**2, 1),
            }
    except ImportError:
        pass
    return {}