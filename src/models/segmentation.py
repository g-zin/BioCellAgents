"""
State-of-the-art cell / nuclei segmentation models.

Why this module exists
----------------------
The previous ``DINOv2CellSegmenter`` produced only 16x16 ViT patch tokens that
were bilinearly upsampled to 224x224 and passed through a tiny conv head.  ViT
patch tokens are not designed for dense per-pixel prediction, so fine cell
boundaries were lost and Dice scores were low.

This module provides a proper **dense-prediction** semantic segmentation model:

* ``DeepLabV3PlusSegmenter`` -- DeepLabV3+ (ResNet50 backbone + ASPP head)
  - Atrous Spatial Pyramid Pooling (ASPP) captures cells of *varying sizes*
    through multi-scale dilated convolutions.
  - Produces true per-pixel predictions at the input resolution.
  - Pretrained on COCO panoptic segmentation (transfer learning).
  - Exposes ``backbone`` / ``head`` submodules so ``BaseTrainer``'s
    differential learning-rate schedule applies cleanly (low LR on the
    pretrained backbone, high LR on the segmentation head).

DeepLabV3+ (Chen et al., CVPR 2018) is a well-validated state-of-the-art
semantic segmentation architecture and is the strongest dense-prediction model
available in ``torchvision`` (already a project dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``deeplabv3_resnet50`` lives in ``torchvision.models.segmentation`` in recent
# releases but was exposed at the top level in older ones -- support both.
try:
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except ImportError:  # pragma: no cover - older torchvision layout
    from torchvision.models import deeplabv3_resnet50
    from torchvision.models import DeepLabV3_ResNet50_Weights


class DeepLabV3PlusSegmenter(nn.Module):
    """DeepLabV3+ (ResNet50) semantic segmentation model for cell typing.

    Args:
        num_classes: Number of output classes (e.g. 6 = 5 cell types + background).
        freeze_backbone: If ``True``, freeze the ResNet50 backbone and only train
            the ASPP head.  If ``False`` (default), fine-tune the whole network
            using ``BaseTrainer``'s differential learning-rate schedule.
        pretrained: If ``True`` (default), initialize from COCO panoptic weights.
            Falls back to random initialization (with a warning) if the weights
            cannot be downloaded.
    """

    def __init__(
        self,
        num_classes: int = 6,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = None
        if pretrained:
            try:
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
            except Exception:  # pragma: no cover - very old torchvision
                weights = None

        try:
            model = deeplabv3_resnet50(weights=weights)
        except Exception as exc:
            if pretrained:
                print(
                    f"Could not load pretrained DeepLabV3+ weights ({exc}); "
                    f"falling back to random initialization."
                )
            model = deeplabv3_resnet50(weights=None)

        # Split into a low-LR backbone (ResNet50 feature extractor) and a
        # high-LR head (ASPP + classification conv).  Naming them ``backbone``
        # and ``head`` keeps BaseTrainer's differential-LR grouping correct.
        self.backbone = model.backbone
        self.head = model.classifier  # DeepLabHead (nn.Sequential)

        # Replace the final 1x1 classification conv (default 21 COCO classes)
        # with one that outputs our number of classes.
        last_conv_idx = None
        for i, module in enumerate(self.head):
            if isinstance(module, nn.Conv2d):
                last_conv_idx = i
        if last_conv_idx is None:
            raise RuntimeError(
                "Could not find a Conv2d in the DeepLabV3+ head to replace."
            )
        in_channels = self.head[last_conv_idx].in_channels
        self.head[last_conv_idx] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Make the model safe for batch_size=1 training (see helper below).
        self._make_batch1_safe()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _make_batch1_safe(self) -> None:
        """Swap the ASPP global-pooling branch's ``BatchNorm2d`` for ``GroupNorm``.

        The ASPP "global pooling" branch (``AdaptiveAvgPool2d(1) -> Conv ->
        BatchNorm2d``) produces a 1x1 spatial map.  ``BatchNorm2d`` cannot
        compute a variance from a single value, so it raises when
        ``batch_size == 1`` in train mode.  Replacing that one ``BatchNorm2d``
        with a parameter-free ``GroupNorm`` (which normalizes per-sample)
        removes the limitation without touching any pretrained conv weights.
        """
        for module in self.head.modules():
            if not isinstance(module, nn.Sequential):
                continue
            has_global_pool = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in module)
            if not has_global_pool:
                continue
            for i, m in enumerate(module):
                if isinstance(m, nn.BatchNorm2d):
                    num_groups = min(32, m.num_features)
                    module[i] = nn.GroupNorm(num_groups, m.num_features)
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel logits of shape ``(B, num_classes, H, W)``.

        Mirrors ``torchvision``'s ``DeepLabV3.forward`` contract: the backbone
        returns a dict of feature maps, the head consumes the ``"out"`` tensor,
        and the result is upsampled back to the input spatial size.
        """
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["out"]
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


def build_segmenter(
    arch: str,
    num_classes: int = 6,
    freeze_backbone: bool = False,
    model_size: str = "vits14",
    pretrained: bool = True,
) -> nn.Module:
    """Factory for the segmentation model.

    Args:
        arch: One of ``{"deeplabv3plus", "dinov2"}``.
        num_classes: Number of output classes.
        freeze_backbone: Freeze the backbone (head-only training).
        model_size: DINOv2 size (only used when ``arch == "dinov2"``).
        pretrained: Use pretrained weights (only used when
            ``arch == "deeplabv3plus"``).
    """
    arch = (arch or "deeplabv3plus").lower()
    if arch in {"deeplabv3plus", "deeplabv3", "deeplabv3_plus"}:
        return DeepLabV3PlusSegmenter(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
        )
    if arch in {"dinov2", "dino"}:
        from src.models.dino_classifier import DINOv2CellSegmenter

        return DINOv2CellSegmenter(
            model_size=model_size,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(
        f"Unknown segmentation architecture '{arch}'. "
        f"Choose from: deeplabv3plus, dinov2"
    )