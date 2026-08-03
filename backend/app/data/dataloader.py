"""
DataLoader Module

Creates PyTorch DataLoaders for training,
validation and testing.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from app.utils.seed import seed_worker, get_generator

DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TRAIN,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=seed_worker,
    generator=get_generator(),
)

from app.core.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SHUFFLE_TRAIN,
)

from app.data.dataset import (
    get_train_dataset,
    get_validation_dataset,
    get_test_dataset,
)

# ==========================================================
# Create Individual DataLoaders
# ==========================================================


def get_train_dataloader() -> DataLoader:
    """
    Returns the training DataLoader.
    """

    train_dataset = get_train_dataset()

    return DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )


def get_validation_dataloader() -> DataLoader:
    """
    Returns the validation DataLoader.
    """

    val_dataset = get_validation_dataset()

    return DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )


def get_test_dataloader() -> DataLoader:
    """
    Returns the test DataLoader.
    """

    test_dataset = get_test_dataset()

    return DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )


# ==========================================================
# Create All DataLoaders
# ==========================================================


def create_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates all project DataLoaders.

    Returns
    -------
    tuple
        (train_loader, validation_loader, test_loader)
    """

    train_loader = get_train_dataloader()

    validation_loader = get_validation_dataloader()

    test_loader = get_test_dataloader()

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ==========================================================
# Dataset Information
# ==========================================================


def get_dataloader_info() -> dict:
    """
    Returns useful information about the DataLoaders.
    """

    train_loader, val_loader, test_loader = create_dataloaders()

    return {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "train_batches": len(train_loader),
        "validation_batches": len(val_loader),
        "test_batches": len(test_loader),
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "classes": train_loader.dataset.classes,
        "num_classes": len(train_loader.dataset.classes),
    }


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    info = get_dataloader_info()

    print("=" * 60)
    print("DataLoader Summary")
    print("=" * 60)

    print(f"Batch Size          : {info['batch_size']}")
    print(f"Workers             : {info['num_workers']}")
    print(f"Classes             : {info['num_classes']}")

    print("-" * 60)

    print(f"Training Images     : {info['train_images']}")
    print(f"Validation Images   : {info['validation_images']}")
    print(f"Testing Images      : {info['test_images']}")

    print("-" * 60)

    print(f"Training Batches    : {info['train_batches']}")
    print(f"Validation Batches  : {info['validation_batches']}")
    print(f"Testing Batches     : {info['test_batches']}")

    print("=" * 60)

    # Verify one batch
    train_loader, _, _ = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("Sample Batch")
    print("-" * 60)
    print(f"Images Shape : {images.shape}")
    print(f"Labels Shape : {labels.shape}")
    print("=" * 60)