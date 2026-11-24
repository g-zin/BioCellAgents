"""Centralized compute device, hardware telemetry, AMP, and safe checkpoint utilities.

Provides a single source of truth for GPU/MPS/CPU resolution, PyTorch 2.7+ safe
checkpoint loading, CUDA AMP autocast, CPU-safe GradScaler creation, and
DataLoader pin_memory decisions across training, inference, evaluation, and UI.
"""

from __future__ import annotations

import contextlib
import logging
import pickle
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def _is_mps_available() -> bool:
    """Return True if Apple Silicon MPS backend is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_available) and is_available())


def _normalize_device(device: Union[torch.device, str]) -> torch.device:
    """Convert a device string or torch.device into a torch.device instance."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device.strip().lower())


def get_device(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the compute device with safe CPU fallback.

    Args:
        preferred: Explicit device preference (e.g. ``"cuda"``, ``"cuda:0"``,
            ``"mps"``, ``"cpu"``, ``torch.device``), or ``None`` / ``"auto"``
            for automatic priority selection (CUDA -> MPS -> CPU).

    Returns:
        Resolved ``torch.device`` guaranteed to be usable on this host.
    """
    if isinstance(preferred, str):
        cleaned = preferred.strip().lower()
        if cleaned in ("", "auto"):
            preferred = None
        else:
            preferred = cleaned

    if preferred is not None:
        requested = _normalize_device(preferred)
        if requested.type == "cuda":
            if torch.cuda.is_available():
                if requested.index is not None and requested.index >= torch.cuda.device_count():
                    logger.warning(
                        "Requested CUDA device '%s' is out of range (count=%d); falling back to CPU.",
                        requested,
                        torch.cuda.device_count(),
                    )
                    return torch.device("cpu")
                return requested
            logger.warning(
                "Requested device '%s' but CUDA is unavailable; falling back to CPU.",
                requested,
            )
            return torch.device("cpu")

        if requested.type == "mps":
            if _is_mps_available():
                return torch.device("mps")
            logger.warning(
                "Requested device '%s' but MPS is unavailable; falling back to CPU.",
                requested,
            )
            return torch.device("cpu")

        return requested

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hardware_info(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """Return a JSON-serializable dictionary of compute hardware telemetry.

    Args:
        device: Optional device preference; auto-detected if ``None``.

    Returns:
        Dictionary containing device identifier, backend availability,
        PyTorch/CUDA versions, GPU name, and VRAM statistics in GiB.
    """
    resolved = get_device(device)
    cuda_available = bool(torch.cuda.is_available())
    mps_available = _is_mps_available()
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = getattr(torch.version, "cuda", None)
    torch_version = str(torch.__version__)
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None

    device_str = str(resolved)
    if resolved.type == "cuda" and cuda_available:
        try:
            idx = resolved.index if resolved.index is not None else int(torch.cuda.current_device())
        except Exception:
            idx = 0
        device_str = f"cuda:{idx}"

        try:
            gpu_name = str(torch.cuda.get_device_name(idx))
        except Exception:
            gpu_name = None

        gb_divisor = float(1024**3)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            vram_free_gb = round(float(free_bytes) / gb_divisor, 3)
            vram_total_gb = round(float(total_bytes) / gb_divisor, 3)
        except Exception:
            try:
                props = torch.cuda.get_device_properties(idx)
                vram_total_gb = round(float(props.total_memory) / gb_divisor, 3)
            except Exception:
                vram_total_gb = None

        try:
            vram_allocated_gb = round(float(torch.cuda.memory_allocated(idx)) / gb_divisor, 3)
        except Exception:
            vram_allocated_gb = None

        try:
            vram_reserved_gb = round(float(torch.cuda.memory_reserved(idx)) / gb_divisor, 3)
        except Exception:
            vram_reserved_gb = None

    return {
        "device": device_str,
        "device_type": resolved.type,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_count": device_count,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "vram_total_gb": vram_total_gb,
        "vram_allocated_gb": vram_allocated_gb,
        "vram_reserved_gb": vram_reserved_gb,
        "vram_free_gb": vram_free_gb,
    }


def load_checkpoint_safe(
    path: Union[str, Path],
    map_location: Union[torch.device, str] = "cpu",
    extract_state_dict: bool = False,
) -> Dict[str, Any]:
    """Load a PyTorch checkpoint safely across PyTorch 2.x (including 2.7+).

    Attempts ``weights_only=True`` first for security, falling back cleanly to
    ``weights_only=False`` when legacy checkpoints include non-tensor metadata
    (e.g. optimizer states, config objects, or scalars blocked by strict unpickling).

    Args:
        path: Path to the checkpoint file.
        map_location: Target device or location passed to ``torch.load``.
        extract_state_dict: If ``True`` and the loaded dict contains
            ``"model_state_dict"``, return that nested state dict directly.

    Returns:
        Loaded checkpoint dictionary (or extracted ``model_state_dict``).

    Raises:
        FileNotFoundError: If ``path`` is empty or does not point to an existing file.
    """
    if not path or not str(path).strip():
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError) as exc:
        if isinstance(exc, TypeError) and "weights_only" in str(exc):
            logger.debug(
                "torch.load rejected weights_only for '%s' (%s); retrying without weights_only.",
                ckpt_path,
                exc,
            )
            checkpoint = torch.load(ckpt_path, map_location=map_location)
        else:
            logger.debug(
                "weights_only=True load failed for '%s' (%s); falling back to weights_only=False.",
                ckpt_path,
                exc,
            )
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if extract_state_dict and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def amp_autocast(
    device: Union[torch.device, str],
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> AbstractContextManager[Any]:
    """Return a CUDA AMP autocast context manager when running on CUDA, else no-op.

    Args:
        device: Active compute device.
        enabled: Whether mixed precision is enabled.
        dtype: Optional autocast dtype (defaults to ``torch.float16`` on CUDA).

    Returns:
        ``torch.amp.autocast`` context manager on CUDA, or ``contextlib.nullcontext()``.
    """
    dev = _normalize_device(device)
    if enabled and dev.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=dtype or torch.float16)
    return contextlib.nullcontext()


def create_grad_scaler(
    device: Union[torch.device, str],
    enabled: bool = True,
) -> Optional[torch.amp.GradScaler]:
    """Create a CUDA ``GradScaler`` only when enabled on an available CUDA device.

    Returns ``None`` on CPU/MPS or when disabled, avoiding PyTorch 2.7+ warnings
    emitted by ``torch.amp.GradScaler("cuda", enabled=False)`` on CPU-only hosts.
    """
    dev = _normalize_device(device)
    if enabled and dev.type == "cuda" and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda", enabled=True)
    return None


def pin_memory_for_device(device: Union[torch.device, str]) -> bool:
    """Return True iff DataLoader ``pin_memory`` should be enabled for ``device``."""
    dev = _normalize_device(device)
    return bool(dev.type == "cuda" and torch.cuda.is_available())


__all__ = [
    "get_device",
    "get_hardware_info",
    "load_checkpoint_safe",
    "amp_autocast",
    "create_grad_scaler",
    "pin_memory_for_device",
]
