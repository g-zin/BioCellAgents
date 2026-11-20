"""Image and label transforms for PanNuke histopathology data."""

from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np


def clean_organ_label(label: Any) -> str:
    """Normalize an organ label loaded from numpy/metadata into a clean string."""
    if label is None:
        return "Unknown"

    if isinstance(label, np.ndarray):
        if label.ndim == 0:
            label = label.item()
        elif label.size > 0:
            label = label.flat[0]
        else:
            return "Unknown"

    if isinstance(label, (bytes, np.bytes_)):
        text = label.decode("utf-8", errors="replace")
    else:
        text = str(label)

    text = text.strip().strip("'\"").strip()
    return text if text else "Unknown"


def crop_nucleus(
    image: np.ndarray,
    centroid: Tuple[float, float],
    crop_size: int = 48,
) -> np.ndarray:
    """Extract a fixed-size square patch centered at (y, x) with border padding."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")

    h, w = image.shape[:2]
    cy = int(round(centroid[0]))
    cx = int(round(centroid[1]))
    half = crop_size // 2
    rem = crop_size - half

    pad_top = max(0, half - cy)
    pad_bottom = max(0, (cy + rem) - h)
    pad_left = max(0, half - cx)
    pad_right = max(0, (cx + rem) - w)

    y1 = max(0, min(h, cy - half))
    y2 = max(0, min(h, cy + rem))
    x1 = max(0, min(w, cx - half))
    x2 = max(0, min(w, cx + rem))

    if y1 >= y2 or x1 >= x2:
        out_shape = (
            (crop_size, crop_size, image.shape[2]) if image.ndim == 3 else (crop_size, crop_size)
        )
        return np.zeros(out_shape, dtype=image.dtype)

    crop = image[y1:y2, x1:x2]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop_h, crop_w = crop.shape[:2]
        border_mode = (
            cv2.BORDER_REFLECT
            if crop_h > max(pad_top, pad_bottom) and crop_w > max(pad_left, pad_right)
            else cv2.BORDER_REPLICATE
        )
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode,
        )
        if image.ndim == 3 and crop.ndim == 2:
            crop = crop[:, :, np.newaxis]

    return crop


def extract_cell_crops(
    image: np.ndarray,
    masks: np.ndarray,
    crop_size: int = 48,
    min_area: int = 10,
) -> List[np.ndarray]:
    """Extract fixed-size RGB crops for each cell instance above ``min_area``."""
    crops: List[np.ndarray] = []

    if masks.ndim == 3:
        channels = [masks[:, :, c] for c in range(min(5, masks.shape[-1]))]
    else:
        channels = [masks]

    for channel_mask in channels:
        instance_ids = np.unique(channel_mask)
        instance_ids = instance_ids[instance_ids != 0]
        for inst_id in instance_ids:
            ys, xs = np.where(channel_mask == inst_id)
            if len(ys) < min_area:
                continue
            centroid = (float(np.mean(ys)), float(np.mean(xs)))
            crops.append(crop_nucleus(image, centroid, crop_size=crop_size))

    return crops


def extract_organ_patches(
    image: np.ndarray,
    patch_size: int = 128,
    stride: int = 128,
) -> List[np.ndarray]:
    """Extract sliding-window patches from an image for organ classification."""
    h, w = image.shape[:2]
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        return [padded[:patch_size, :patch_size]]

    patches: List[np.ndarray] = []
    y_starts = list(range(0, h - patch_size + 1, max(1, stride)))
    x_starts = list(range(0, w - patch_size + 1, max(1, stride)))

    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)

    for y in y_starts:
        for x in x_starts:
            patches.append(image[y : y + patch_size, x : x + patch_size])

    return patches
