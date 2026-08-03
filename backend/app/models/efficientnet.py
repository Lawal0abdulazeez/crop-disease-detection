"""
EfficientNet Model

Creates an EfficientNet-B0 classifier for
PlantVillage disease classification.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from app.core.config import (
    PRETRAINED,
    FREEZE_BACKBONE,
    DROPOUT,
)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 classifier.
    """

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if PRETRAINED
            else None
        )

        self.model = efficientnet_b0(weights=weights)

        if FREEZE_BACKBONE:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def create_efficientnet(num_classes: int):
    """
    Factory helper.
    """
    return EfficientNetClassifier(num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict:
    """
    Count total, trainable and frozen parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def model_summary(
    model: nn.Module,
    model_name: str,
    num_classes: int,
    image_size: int = 224,
) -> None:
    """
    Print a concise model summary.
    """
    stats = count_parameters(model)

    print("\n")
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Architecture        : {model_name}")
    print(f"Input Size          : 3 x {image_size} x {image_size}")
    print(f"Output Classes      : {num_classes}")
    print()
    print(f"Total Parameters    : {stats['total']:,}")
    print(f"Trainable Params    : {stats['trainable']:,}")
    print(f"Frozen Parameters   : {stats['frozen']:,}")
    print()
    print(f"Pretrained          : {PRETRAINED}")
    print(f"Backbone Frozen     : {FREEZE_BACKBONE}")
    print(f"Dropout             : {DROPOUT}")
    print("=" * 70)


if __name__ == "__main__":
    model = create_efficientnet(num_classes=15)
    model_summary(
        model=model,
        model_name="efficientnet_b0",
        num_classes=15,
    )
