"""Cellpose-based cell segmentation and semantic segmentation adapter.

This module provides two segmenter classes that expose a unified
``.segment(image_pil) -> np.ndarray`` interface, returning a binary mask
(uint8, 0/1) at the original image resolution:

- ``CellposeSegmenter``: Zero-shot instance segmentation using Cellpose-SAM v2.
  No training required -- works out of the box on H&E tissue images.

- ``SemanticSegmenterAdapter``: Wraps a torch semantic segmentation model
  (e.g. DeepLabV3+) and converts its per-pixel logits into a binary mask.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.utils.device import get_device


def _build_cellpose_model(cellpose_cls: Any, pretrained_model: str, device: torch.device) -> Any:
    """Instantiate ``CellposeModel`` compatibly across Cellpose v3 and v4."""
    use_gpu = device.type == "cuda"
    try:
        params = inspect.signature(cellpose_cls.__init__).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict[str, Any] = {}
        if "pretrained_model" in params or has_var_kw:
            kwargs["pretrained_model"] = pretrained_model
        if "device" in params:
            kwargs["device"] = device
        if "gpu" in params:
            kwargs["gpu"] = use_gpu
        if has_var_kw and "device" not in params and "gpu" not in params:
            kwargs["gpu"] = use_gpu
            kwargs["device"] = device
        return cellpose_cls(**kwargs)
    except (TypeError, ValueError):
        for fallback_kwargs in (
            {"gpu": use_gpu, "pretrained_model": pretrained_model},
            {"pretrained_model": pretrained_model, "device": device},
            {"pretrained_model": pretrained_model},
        ):
            try:
                return cellpose_cls(**fallback_kwargs)
            except TypeError:
                continue
        return cellpose_cls()


class CellposeSegmenter:
    """Cellpose-based cell segmentation (zero-shot, no training required).

    Wraps Cellpose-SAM v2 for instance segmentation of cells in H&E tissue images.
    Exposes a unified ``.segment(image_pil) -> np.ndarray`` interface.

    Args:
        pretrained_model: Cellpose model variant. Default ``"cpsam_v2"`` (Cellpose-SAM v2).
        device: Torch device or string (e.g. ``"cuda"``, ``"cpu"``). Auto-detected if ``None``.
        model: Optional pre-instantiated Cellpose-compatible model (used for offline testing).
    """

    def __init__(
        self,
        pretrained_model: str = "cpsam_v2",
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[Any] = None,
    ):
        self.device = get_device(device)
        self.pretrained_model = pretrained_model

        if model is not None:
            self.model = model
        else:
            try:
                from cellpose.models import CellposeModel
            except ImportError as exc:
                raise ImportError(
                    "cellpose is required for CellposeSegmenter. "
                    "Install it with: pip install cellpose"
                ) from exc

            self.model = _build_cellpose_model(CellposeModel, pretrained_model, self.device)
        print(f"CellposeSegmenter initialized (model={pretrained_model}, device={self.device})")

    def _run_eval(self, img_array: np.ndarray) -> Any:
        """Invoke ``self.model.eval`` compatibly across Cellpose v3 and v4 signatures."""
        try:
            params: dict[str, inspect.Parameter] = dict(
                inspect.signature(self.model.eval).parameters
            )
        except (TypeError, ValueError):
            params = {}

        try:
            if "channel_axis" in params:
                if img_array.ndim == 3:
                    return self.model.eval(img_array, channel_axis=-1)
                return self.model.eval(img_array)
            if "channels" in params:
                return self.model.eval(img_array, channels=[0, 0])
            return self.model.eval(img_array)
        except TypeError:
            return self.model.eval(img_array)

    def segment(self, image_pil: Image.Image) -> np.ndarray:
        """Segment cells in the image.

        Args:
            image_pil: PIL Image (RGB, RGBA, or grayscale).

        Returns:
            Binary mask of shape (H, W) with dtype uint8, where 1 = cell, 0 = background.
        """
        if image_pil.mode not in ("RGB", "L"):
            image_pil = image_pil.convert("RGB")
        img_array = np.array(image_pil)

        result = self._run_eval(img_array)
        # Unpack safely whether Cellpose returns a 3-tuple (v4) or 4-tuple (legacy v2/v3)
        masks = result[0] if isinstance(result, (tuple, list)) else result

        # Convert instance mask to binary mask
        binary_mask = (np.asarray(masks) > 0).astype(np.uint8)
        return binary_mask


class SemanticSegmenterAdapter:
    """Adapter that wraps a torch semantic segmentation model to expose the
    unified ``.segment(image_pil) -> np.ndarray`` interface.

    Args:
        model: A torch ``nn.Module`` that takes a ``(B, 3, 224, 224)`` tensor and
            returns logits of shape ``(B, num_classes, H, W)``.
        device: Torch device or string. Auto-detected if ``None``.
    """

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.device = get_device(device)
        self.model = model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def segment(self, image_pil: Image.Image) -> np.ndarray:
        """Segment cells using the wrapped semantic model.

        Args:
            image_pil: PIL Image (RGB or convertible to RGB).

        Returns:
            Binary mask of shape (H, W) with dtype uint8, where 1 = cell, 0 = background.
        """
        image_pil = image_pil.convert("RGB")
        orig_w, orig_h = image_pil.size

        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        seg_logits = self.model(input_tensor)  # (1, num_classes, 224, 224)

        # Map multi-class logits to binary mask (class > 0 = foreground)
        pred_class_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        binary_mask_224 = (pred_class_mask > 0).astype(np.uint8)

        # Resize to original resolution
        binary_mask = cv2.resize(binary_mask_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return binary_mask
