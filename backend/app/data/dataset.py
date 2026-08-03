"""
Dataset Loader

Loads the train, validation and test datasets
using torchvision.datasets.ImageFolder.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import json
from pathlib import Path

from torchvision.datasets import ImageFolder

from app.core.config import (
    SPLIT_DATA_DIR,
    METADATA_DIR,
    CLASS_NAMES_FILE,
)

from app.data.transforms import (
    train_transform,
    val_transform,
    test_transform,
)

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = SPLIT_DATA_DIR / "train"

VAL_DIR = SPLIT_DATA_DIR / "val"

TEST_DIR = SPLIT_DATA_DIR / "test"


# ==========================================================
# Validation
# ==========================================================

def validate_dataset_structure() -> None:
    """
    Ensure the dataset split folders exist.
    """

    required_dirs = [
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ]

    for directory in required_dirs:

        if not directory.exists():

            raise FileNotFoundError(
                f"\nDataset folder not found:\n{directory}\n"
                "Run:\n"
                "uv run python scripts/prepare_dataset.py"
            )


# ==========================================================
# Save Class Names
# ==========================================================

def save_class_names(dataset: ImageFolder) -> None:
    """
    Save class names for inference.

    The saved JSON file will later be used by FastAPI
    to convert predicted class indices back into
    disease names.
    """

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as file:

        json.dump(
            dataset.classes,
            file,
            indent=4,
        )


# ==========================================================
# Dataset Loaders
# ==========================================================

def get_train_dataset() -> ImageFolder:

    validate_dataset_structure()

    dataset = ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform,
    )

    save_class_names(dataset)

    return dataset


def get_validation_dataset() -> ImageFolder:

    validate_dataset_structure()

    return ImageFolder(
        root=str(VAL_DIR),
        transform=val_transform,
    )


def get_test_dataset() -> ImageFolder:

    validate_dataset_structure()

    return ImageFolder(
        root=str(TEST_DIR),
        transform=test_transform,
    )


# ==========================================================
# Utility
# ==========================================================

def get_class_names() -> list[str]:
    """
    Return the dataset class names.
    """

    dataset = get_train_dataset()

    return dataset.classes


def get_num_classes() -> int:
    """
    Return the total number of classes.
    """

    return len(get_class_names())


def get_dataset_summary() -> dict:
    """
    Return a summary of the datasets.
    """

    train_dataset = get_train_dataset()

    val_dataset = get_validation_dataset()

    test_dataset = get_test_dataset()

    summary = {

        "num_classes": len(train_dataset.classes),

        "train_samples": len(train_dataset),

        "validation_samples": len(val_dataset),

        "test_samples": len(test_dataset),

        "classes": train_dataset.classes,
    }

    return summary


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    summary = get_dataset_summary()

    print("=" * 60)

    print("Dataset Summary")

    print("=" * 60)

    print(f"Classes            : {summary['num_classes']}")

    print(f"Training Images    : {summary['train_samples']}")

    print(f"Validation Images  : {summary['validation_samples']}")

    print(f"Testing Images     : {summary['test_samples']}")

    print("\nClass Names")

    print("-" * 60)

    for index, name in enumerate(summary["classes"]):

        print(f"{index:02d} -> {name}")

    print("=" * 60)