import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.device import amp_autocast

os.environ["XFORMERS_DISABLED"] = "1"


class DINOv2CellClassifier(nn.Module):
    """STEP 3: Classification model using DINOv2 backbone for single-cell crop typing."""

    def __init__(
        self,
        model_size: str = "vits14",
        num_classes: int = 5,
        freeze_backbone: bool = True,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        embed_dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024}
        if model_size not in embed_dims:
            raise ValueError(
                f"Unknown model_size: {model_size}. Choose from {list(embed_dims.keys())}"
            )

        embed_dim = embed_dims[model_size]
        self.backbone: Any = (
            backbone
            if backbone is not None
            else torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_size}")
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification. Returns logits (B x num_classes)."""
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        with amp_autocast(x.device):
            features = self.backbone(x)

        features = features.to(x.dtype)
        return self.classifier_head(features)

    def classify_crops(
        self,
        crops: Union[torch.Tensor, np.ndarray, Sequence[Union[torch.Tensor, np.ndarray]]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Classify a list, 3D crop, or 4D batch of cell crops in batches."""
        crop_list: List[Union[torch.Tensor, np.ndarray]]
        if isinstance(crops, (torch.Tensor, np.ndarray)):
            if crops.ndim == 3:
                crop_list = [crops]
            elif crops.ndim == 4:
                if crops.shape[0] == 0:
                    return []
                crop_list = [crops[i] for i in range(crops.shape[0])]
            else:
                raise ValueError(
                    f"Expected 3D or 4D tensor/array for crops, got {crops.ndim}D "
                    f"with shape {tuple(crops.shape)}"
                )
        else:
            if len(crops) == 0:
                return []
            crop_list = list(crops)

        self.eval()
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        results: List[Dict[str, Any]] = []

        with torch.no_grad():
            for i in range(0, len(crop_list), batch_size):
                batch = crop_list[i : i + batch_size]
                normalized_batch: List[torch.Tensor] = []
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        t = item.detach().to(dtype=torch.float32)
                    else:
                        t = torch.as_tensor(np.asarray(item), dtype=torch.float32)
                    if t.ndim == 4 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
                        t = t.permute(2, 0, 1)
                    normalized_batch.append(t.to(model_device))

                batch_tensor = torch.stack(normalized_batch, dim=0)

                logits = self.forward(batch_tensor)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1).values

                for j, (logit, pred, conf) in enumerate(zip(logits, preds, confs)):
                    results.append(
                        {
                            "batch_idx": i + j,
                            "logits": logit.cpu().numpy(),
                            "predicted_class": int(pred.item()),
                            "confidence": float(conf.item()),
                        }
                    )

        return results


class DINOv2CellSegmenter(nn.Module):
    """DINOv2-based cell segmentation model.

    Uses a DINOv2 backbone with a lightweight convolutional decoder head
    to produce per-pixel class predictions for nuclei segmentation.
    """

    def __init__(
        self,
        model_size: str = "vits14",
        num_classes: int = 6,
        freeze_backbone: bool = True,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        embed_dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024}
        if model_size not in embed_dims:
            raise ValueError(
                f"Unknown model_size: {model_size}. Choose from {list(embed_dims.keys())}"
            )

        embed_dim = embed_dims[model_size]
        self.backbone: Any = (
            backbone
            if backbone is not None
            else torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_size}")
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation. Returns per-pixel logits (B x C x H x W)."""
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        with amp_autocast(x.device):
            out = self.backbone.forward_features(x)

        # DINOv2's forward_features returns a dict; extract the normalized patch
        # tokens (B, N, D) which form the spatial feature map.
        if isinstance(out, dict):
            spatial = out.get("x_norm_patchtokens")
            if spatial is None:
                spatial = out.get("x_prenorm")
        else:
            spatial = out

        if spatial is None:
            raise RuntimeError("DINOv2 forward_features returned no spatial tokens")

        if spatial.dim() == 3:
            B, N, D = spatial.shape
            H = W = int(N**0.5)
            if H * W == 0:
                raise RuntimeError("DINOv2 forward_features returned empty spatial tokens")
            if H * W != N:
                spatial = spatial[:, -H * W :, :]
            spatial = spatial.permute(0, 2, 1).reshape(B, D, H, W)
        elif spatial.dim() == 2:
            B, D = spatial.shape
            H = W = 224 // 14
            spatial = spatial.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

        spatial = spatial.to(x.dtype)
        spatial = F.interpolate(spatial, size=(224, 224), mode="bilinear", align_corners=False)

        logits = self.decoder(spatial)
        return logits
