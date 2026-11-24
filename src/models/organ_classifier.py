from typing import Tuple, Union

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0 as efficientnet


class OrganClassifier(nn.Module):
    """EfficientNet-B0 backbone fine-tuned for 19-class pathology organ identification.

    Includes support for backbone freezing, feature embedding extraction,
    and direct access to weight-associated transforms.
    """

    def __init__(
        self,
        num_classes: int = 19,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = efficientnet(weights=weights)
        except Exception as exc:
            if pretrained:
                print(
                    f"Could not load pretrained EfficientNet-B0 weights ({exc}); "
                    f"falling back to random initialization."
                )
            self.backbone = efficientnet(weights=None)
        # Capture the feature dimension from the original classifier head
        # (nn.Sequential(Dropout, Linear)) BEFORE replacing it with Identity.
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def get_transforms(self):
        """Returns the standard ImageNet pre-processing transforms for this backbone."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def unfreeze_backbone(self, unfreeze: bool = True):
        """Dynamically toggles grad status for fine-tuning backbones post-warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = unfreeze

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through EfficientNet-B0 backbone and classification head.

        Args:
            x: Tensor of shape (Batch, 3, Height, Width)
            return_features: If True, returns tuple of (logits, embedding)

        Returns:
            Logits of shape (Batch, num_classes) or tuple of (logits, features)
        """
        features = self.backbone(x)  # (Batch, embedding_dim)
        logits = self.classifier(features)  # (Batch, num_classes)

        if return_features:
            return logits, features
        return logits
