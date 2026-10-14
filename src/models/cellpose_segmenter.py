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

import numpy as np
import cv2
from PIL import Image
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms


class CellposeSegmenter:
    """Cellpose-based cell segmentation (zero-shot, no training required).

    Wraps Cellpose-SAM v2 for instance segmentation of cells in H&E tissue images.
    Exposes a unified ``.segment(image_pil) -> np.ndarray`` interface.

    Args:
        pretrained_model: Cellpose model variant. Default ``"cpsam_v2"`` (Cellpose-SAM v2).
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``). Auto-detected if ``None``.
    """

    def __init__(self, pretrained_model: str = "cpsam_v2", device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            from cellpose.models import CellposeModel
        except ImportError as exc:
            raise ImportError(
                "cellpose is required for CellposeSegmenter. "
                "Install it with: pip install cellpose"
            ) from exc

        self.model = CellposeModel(pretrained_model=pretrained_model, device=self.device)
        print(f"CellposeSegmenter initialized (model={pretrained_model}, device={self.device})")

    def segment(self, image_pil: Image.Image) -> np.ndarray:
        """Segment cells in the image.

        Args:
            image_pil: PIL Image (RGB or grayscale).

        Returns:
            Binary mask of shape (H, W) with dtype uint8, where 1 = cell, 0 = background.
        """
        img_array = np.array(image_pil)

        # Cellpose v4: eval() with channels=[0, 0] uses first channel for both
        masks, flows, styles = self.model.eval(
            img_array,
            channels=[0, 0],
        )

        # Convert instance mask to binary mask
        binary_mask = (masks > 0).astype(np.uint8)
        return binary_mask


class SemanticSegmenterAdapter:
    """Adapter that wraps a torch semantic segmentation model to expose the
    unified ``.segment(image_pil) -> np.ndarray`` interface.

    Args:
        model: A torch ``nn.Module`` that takes a ``(B, 3, 224, 224)`` tensor and
            returns logits of shape ``(B, num_classes, H, W)``.
        device: Torch device string. Auto-detected if ``None``.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def segment(self, image_pil: Image.Image) -> np.ndarray:
        """Segment cells using the wrapped semantic model.

        Args:
            image_pil: PIL Image (RGB).

        Returns:
            Binary mask of shape (H, W) with dtype uint8, where 1 = cell, 0 = background.
        """
        orig_w, orig_h = image_pil.size

        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        seg_logits = self.model(input_tensor)  # (1, num_classes, 224, 224)

        # Map multi-class logits to binary mask (class > 0 = foreground)
        pred_class_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        binary_mask_224 = (pred_class_mask > 0).astype(np.uint8)

        # Resize to original resolution
        binary_mask = cv2.resize(
            binary_mask_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        return binary_mask