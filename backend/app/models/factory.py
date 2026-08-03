"""
Model Factory

Centralized model creation for the Crop Disease Detection project.

Every training, evaluation and inference script should create
models from this module instead of directly instantiating a
specific architecture.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch.nn as nn

from app.core.config import MODEL_NAME

from app.models.efficientnet import (
    create_efficientnet,
    model_summary,
)


SUPPORTED_MODELS = (
    "efficientnet_b0",
)


def create_model(
    num_classes: int,
    model_name: str | None = None,
    show_summary: bool = True,
) -> nn.Module:
    """
    Create a model.

    Parameters
    ----------
    num_classes
        Number of output classes.

    model_name
        Override config model.

    show_summary
        Print model information.
    """

    model_name = model_name or MODEL_NAME

    model_name = model_name.lower()

    if model_name == "efficientnet_b0":

        model = create_efficientnet(
            num_classes=num_classes,
        )

    else:

        supported = ", ".join(SUPPORTED_MODELS)

        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {supported}"
        )

    if show_summary:

        model_summary(
            model=model,
            model_name=model_name,
            num_classes=num_classes,
        )

    return model


def list_models() -> list[str]:
    """
    Return all supported models.
    """

    return list(SUPPORTED_MODELS)


if __name__ == "__main__":

    print("Supported Models")

    print("----------------")

    for model in list_models():

        print(model)

    print()

    model = create_model(
        num_classes=15,
    )