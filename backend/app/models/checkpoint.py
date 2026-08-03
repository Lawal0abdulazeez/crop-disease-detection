"""
Checkpoint Utilities

Handles saving and loading model checkpoints.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from app.core.config import (
    CHECKPOINT_DIR,
    SAVE_LAST,
    SAVE_BEST,
    SAVE_EVERY_N_EPOCHS,
    BEST_MODEL_NAME,
    LAST_MODEL_NAME,
)

# ==========================================================
# Initialization
# ==========================================================


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Save Checkpoint
# ==========================================================


def save_checkpoint(
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    train_loss: float,
    val_loss: float,
    train_accuracy: float,
    val_accuracy: float,
    model_name: str,
    num_classes: int,
    is_best: bool = False,
):
    """
    Save model checkpoint.
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_name": model_name,
        "num_classes": num_classes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }

    # ----------------------------------------
    # Save Last
    # ----------------------------------------

    if SAVE_LAST:

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / LAST_MODEL_NAME,
        )

    # ----------------------------------------
    # Save Best
    # ----------------------------------------

    if SAVE_BEST and is_best:

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / BEST_MODEL_NAME,
        )

    # ----------------------------------------
    # Save Periodic
    # ----------------------------------------

    if (
        SAVE_EVERY_N_EPOCHS > 0
        and epoch % SAVE_EVERY_N_EPOCHS == 0
    ):

        filename = (
            f"epoch_{epoch:03d}.pt"
        )

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / filename,
        )

    return epoch_checkpoint


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """
    Load a checkpoint from disk.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict")
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict")
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint

# ==========================================================
# Latest Checkpoint
# ==========================================================


def latest_checkpoint() -> Path | None:
    """
    Return last_model.pt if it exists.
    """

    path = CHECKPOINT_DIR / LAST_MODEL_NAME

    if path.exists():

        return path

    return None


# ==========================================================
# Best Checkpoint
# ==========================================================


def best_checkpoint() -> Path | None:
    """
    Return best_model.pt if it exists.
    """

    path = CHECKPOINT_DIR / BEST_MODEL_NAME

    if path.exists():

        return path

    return None


# ==========================================================
# Print Information
# ==========================================================


def print_checkpoint_info(
    checkpoint: dict,
) -> None:

    print()

    print("=" * 70)

    print("CHECKPOINT")

    print("=" * 70)

    print(
        f"Epoch               : {checkpoint['epoch']}"
    )

    print(
        f"Training Loss       : {checkpoint['train_loss']:.4f}"
    )

    print(
        f"Validation Loss     : {checkpoint['validation_loss']:.4f}"
    )

    print(
        f"Training Accuracy   : {checkpoint['train_accuracy']:.2f}%"
    )

    print(
        f"Validation Accuracy : {checkpoint['validation_accuracy']:.2f}%"
    )

    print("=" * 70)