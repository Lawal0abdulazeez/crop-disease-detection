"""
Seed Utilities

Provides reproducible experiments by fixing
random seeds across Python, NumPy and PyTorch.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from app.core.config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader workers.
    """

    worker_seed = RANDOM_SEED + worker_id

    random.seed(worker_seed)

    np.random.seed(worker_seed)


def get_generator() -> torch.Generator:
    """
    Generator for deterministic DataLoader shuffling.
    """

    generator = torch.Generator()

    generator.manual_seed(RANDOM_SEED)

    return generator


if __name__ == "__main__":

    seed_everything()

    print(torch.rand(3))