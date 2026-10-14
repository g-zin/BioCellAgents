import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class DINOv2CellClassifier(nn.Module):
    """STEP 3: Classification model using DINOv2 backbone for single-cell crop typing."""

    def __init__(self, model_size: str = "vits14", num_classes: int = 5, freeze_backbone: bool = True):
        super().__init__()
        embed_dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024}
        if model_size not in embed_dims:
            raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(embed_dims.keys())}")

        embed_dim = embed_dims[model_size]
        self.backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_size}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification. Returns logits (B x num_classes)."""
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        device_type = x.device.type
        if device_type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        features = features.to(x.dtype)
        return self.classifier_head(features)

    def classify_crops(self, crops: List[torch.Tensor], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Classify a list of cell crops in batches."""
        self.eval()
        results = []
        
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = crops[i:i+batch_size]
                if isinstance(batch[0], torch.Tensor):
                    batch_tensor = torch.stack(batch)
                else:
                    batch_tensor = batch
                
                if batch_tensor.device.type == "cuda":
                    batch_tensor = batch_tensor.cuda()
                
                logits = self.forward(batch_tensor)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1).values
                
                for j, (logit, pred, conf) in enumerate(zip(logits, preds, confs)):
                    results.append({
                        "batch_idx": i + j,
                        "logits": logit.cpu().numpy(),
                        "predicted_class": int(pred.item()),
                        "confidence": float(conf.item())
                    })
        
        return results


class DINOv2CellSegmenter(nn.Module):
    """DINOv2-based cell segmentation model.
    
    Uses a DINOv2 backbone with a lightweight convolutional decoder head
    to produce per-pixel class predictions for nuclei segmentation.
    """

    def __init__(self, model_size: str = "vits14", num_classes: int = 6, freeze_backbone: bool = True):
        super().__init__()
        embed_dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024}
        if model_size not in embed_dims:
            raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(embed_dims.keys())}")

        embed_dim = embed_dims[model_size]
        self.backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_size}")

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

        device_type = x.device.type
        if device_type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = self.backbone.forward_features(x)
        else:
            out = self.backbone.forward_features(x)

        # DINOv2's forward_features returns a dict; extract the normalized patch
        # tokens (B, N, D) which form the spatial feature map.
        if isinstance(out, dict):
            spatial = out.get("x_norm_patchtokens", out.get("x_prenorm"))
        else:
            spatial = out

        if spatial.dim() == 3:
            B, N, D = spatial.shape
            H = W = int(N ** 0.5)
            spatial = spatial.permute(0, 2, 1).reshape(B, D, H, W)
        elif spatial.dim() == 2:
            B, D = spatial.shape
            H = W = 224 // 14
            spatial = spatial.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

        spatial = spatial.to(x.dtype)
        spatial = F.interpolate(spatial, size=(224, 224), mode="bilinear", align_corners=False)

        logits = self.decoder(spatial)
        return logits
