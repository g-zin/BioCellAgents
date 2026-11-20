"""Offline CPU unit tests for src.models.segmentation and SemanticSegmenterAdapter."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.cellpose_segmenter import SemanticSegmenterAdapter
from src.models.segmentation import DeepLabV3PlusSegmenter, build_segmenter


class _DummySegNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = torch.zeros(b, 3, h, w, device=x.device)
        logits[:, 1, : h // 2, : w // 2] = 5.0
        return logits


class TestSegmentationModels:
    """Behavior tests for segmentation architectures and adapters (no downloads)."""

    def test_deeplabv3plus_batch1_train_mode_and_shape(self):
        model = DeepLabV3PlusSegmenter(num_classes=6, freeze_backbone=True, pretrained=False)
        model.train()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 6, 64, 64)
        assert all(not p.requires_grad for p in model.backbone.parameters())

    def test_build_segmenter_dispatch_and_invalid_arch(self):
        model = build_segmenter("deeplabv3plus", num_classes=4, pretrained=False)
        assert isinstance(model, DeepLabV3PlusSegmenter)
        assert model.num_classes == 4

        with pytest.raises(ValueError, match="Unknown segmentation architecture"):
            build_segmenter("nonexistent_arch")

    def test_semantic_segmenter_adapter_output_contract(self):
        adapter = SemanticSegmenterAdapter(_DummySegNet(), device="cpu")
        img = Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8))
        mask = adapter.segment(img)
        assert mask.shape == (100, 120)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) > 0
