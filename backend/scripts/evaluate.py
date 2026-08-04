"""
Model Evaluation Script

Loads best (or last) checkpoint and evaluates on the test set.
Works with smoke / debug / full checkpoints interchangeably.

Usage
-----
  uv run python -m scripts.evaluate
  uv run python -m scripts.evaluate --checkpoint last
  uv run python -m scripts.evaluate --checkpoint path/to/custom.pt

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.core.config import (
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LAST_MODEL_NAME,
    MODEL_NAME,
    RANDOM_SEED,
)
from app.core.paths import initialize_project
from app.data.dataloader import get_test_dataloader
from app.data.dataset import get_class_names, get_num_classes
from app.models.factory import create_model
from app.models.checkpoint import load_checkpoint, print_checkpoint_info
from app.training.evaluator import ModelEvaluator
from app.utils.device import get_device
from app.utils.logger import get_logger
from app.utils.seed import seed_everything

logger = get_logger()


def resolve_checkpoint(choice: str) -> Path:
    """
    Resolve checkpoint path from 'best', 'last', or a file path.
    """
    if choice.lower() == "best":
        path = CHECKPOINT_DIR / BEST_MODEL_NAME
    elif choice.lower() == "last":
        path = CHECKPOINT_DIR / LAST_MODEL_NAME
    else:
        path = Path(choice)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train first with: uv run python -m scripts.train"
        )
    return path


def main(checkpoint_choice: str = "best") -> dict:
    logger.info("=" * 70)
    logger.info("Crop Disease Detection — Evaluation")
    logger.info("=" * 70)

    initialize_project(verbose=False)
    seed_everything(RANDOM_SEED)

    device = get_device()
    logger.info(f"Device: {device}")

    checkpoint_path = resolve_checkpoint(checkpoint_choice)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    raw = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    num_classes = raw.get("num_classes") or get_num_classes()
    model_name = raw.get("model_name") or MODEL_NAME

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        show_summary=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
    )
    print_checkpoint_info(checkpoint)

    class_names = get_class_names()
    test_loader = get_test_dataloader()

    evaluator = ModelEvaluator(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        device=device,
    )

    results = evaluator.run()

    logger.info("Evaluation complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'last', or path to a .pt file (default: best)",
    )
    args = parser.parse_args()
    main(checkpoint_choice=args.checkpoint)
