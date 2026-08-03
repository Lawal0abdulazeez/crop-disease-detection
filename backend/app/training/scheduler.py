"""
Learning Rate Scheduler Factory

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    ExponentialLR,
)

from app.core.config import (
    SCHEDULER,
    STEP_SIZE,
    STEP_GAMMA,
    COSINE_T_MAX,
    ETA_MIN,
    REDUCE_FACTOR,
    REDUCE_PATIENCE,
    EXPONENTIAL_GAMMA,
)


SUPPORTED_SCHEDULERS = (
    "none",
    "step",
    "cosine",
    "reduce_on_plateau",
    "exponential",
)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str | None = None,
):
    """
    Create a learning-rate scheduler.
    """

    scheduler_name = (scheduler_name or SCHEDULER).lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "step":

        return StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=STEP_GAMMA,
        )

    if scheduler_name == "cosine":

        return CosineAnnealingLR(
            optimizer,
            T_max=COSINE_T_MAX,
            eta_min=ETA_MIN,
        )

    if scheduler_name == "reduce_on_plateau":

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=REDUCE_FACTOR,
            patience=REDUCE_PATIENCE,
        )

    if scheduler_name == "exponential":

        return ExponentialLR(
            optimizer,
            gamma=EXPONENTIAL_GAMMA,
        )

    supported = ", ".join(SUPPORTED_SCHEDULERS)

    raise ValueError(
        f"Unsupported scheduler '{scheduler_name}'. "
        f"Supported: {supported}"
    )


def scheduler_requires_metric(
    scheduler,
) -> bool:
    """
    Returns True if scheduler.step()
    requires a validation metric.
    """

    return isinstance(
        scheduler,
        ReduceLROnPlateau,
    )


def list_schedulers():

    return list(SUPPORTED_SCHEDULERS)


if __name__ == "__main__":

    import torch

    model = torch.nn.Linear(10, 2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    scheduler = create_scheduler(optimizer)

    print("=" * 60)
    print("Scheduler")
    print("=" * 60)

    print(type(scheduler).__name__)

    print()

    print("Supported")

    for s in list_schedulers():
        print("-", s)

    print("=" * 60)