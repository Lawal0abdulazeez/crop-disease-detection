"""
Optimizer Factory

Creates optimizers for model training.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch.optim as optim
from torch.nn import Module

from app.core.config import (
    OPTIMIZER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MOMENTUM,
    NESTEROV,
)

SUPPORTED_OPTIMIZERS = (
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
)


def create_optimizer(
    model: Module,
    optimizer_name: str | None = None,
):
    """
    Create optimizer for trainable parameters only.
    """

    optimizer_name = (optimizer_name or OPTIMIZER).lower()

    parameters = filter(
        lambda p: p.requires_grad,
        model.parameters(),
    )

    if optimizer_name == "adam":

        return optim.Adam(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if optimizer_name == "adamw":

        return optim.AdamW(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if optimizer_name == "sgd":

        return optim.SGD(
            parameters,
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=NESTEROV,
        )

    if optimizer_name == "rmsprop":

        return optim.RMSprop(
            parameters,
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    supported = ", ".join(SUPPORTED_OPTIMIZERS)

    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. "
        f"Supported: {supported}"
    )


def list_optimizers():
    """
    Return supported optimizers.
    """

    return list(SUPPORTED_OPTIMIZERS)


if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(128, 15)

    optimizer = create_optimizer(model)

    print("=" * 60)
    print("OPTIMIZER")
    print("=" * 60)

    print(type(optimizer).__name__)

    print()

    print("Supported:")

    for opt in list_optimizers():
        print(f"- {opt}")

    print("=" * 60)