"""
Loss Functions

Centralized loss function factory.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch.nn as nn

from app.core.config import LOSS_FUNCTION


SUPPORTED_LOSSES = (
    "cross_entropy",
)


def create_loss(
    loss_name: str | None = None,
) -> nn.Module:
    """
    Create a loss function.

    Parameters
    ----------
    loss_name:
        Optional override of LOSS_FUNCTION from config.

    Returns
    -------
    nn.Module
    """

    loss_name = (loss_name or LOSS_FUNCTION).lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    supported = ", ".join(SUPPORTED_LOSSES)

    raise ValueError(
        f"Unsupported loss '{loss_name}'. "
        f"Supported losses: {supported}"
    )


def list_losses() -> list[str]:
    """
    Return supported loss names.
    """

    return list(SUPPORTED_LOSSES)


if __name__ == "__main__":

    criterion = create_loss()

    print("=" * 60)
    print("LOSS FUNCTION")
    print("=" * 60)

    print(type(criterion).__name__)

    print()

    print("Supported:")

    for loss in list_losses():
        print(f"- {loss}")

    print("=" * 60)