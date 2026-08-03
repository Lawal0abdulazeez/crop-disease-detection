"""
Dataset Splitter

Reads the raw PlantVillage dataset and creates
train/validation/test folders.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from app.core.config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    METADATA_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ==========================================================
# Helpers
# ==========================================================

random.seed(RANDOM_SEED)


def is_image(file_path: Path) -> bool:
    """
    Check whether a file is a supported image.
    """
    return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def copy_images(files: list[Path], destination: Path) -> None:
    """
    Copy image files to destination.
    """
    destination.mkdir(parents=True, exist_ok=True)

    for image_path in files:
        shutil.copy2(image_path, destination / image_path.name)


def clear_previous_split() -> None:
    """
    Remove an existing split directory.
    """
    if SPLIT_DATA_DIR.exists():
        shutil.rmtree(SPLIT_DATA_DIR)

    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Main Split Function
# ==========================================================

def split_dataset() -> None:
    """
    Split dataset into train / val / test folders.
    """

    # ------------------------------------------------------
    # Locate dataset
    # ------------------------------------------------------

    dataset_root = None

    for directory in RAW_DATA_DIR.iterdir():
        if directory.is_dir():
            dataset_root = directory
            break

    if dataset_root is None:
        raise FileNotFoundError(
            f"No dataset folder found inside:\n{RAW_DATA_DIR}"
        )

    print(f"\nDataset Found:\n{dataset_root}\n")

    clear_previous_split()

    metadata = defaultdict(dict)

    total_images = 0

    class_folders = sorted(
        [
            folder
            for folder in dataset_root.iterdir()
            if folder.is_dir()
        ]
    )

    print(f"Found {len(class_folders)} classes.\n")

    # ------------------------------------------------------
    # Process each class
    # ------------------------------------------------------

    for class_folder in class_folders:

        images = [
            image
            for image in class_folder.iterdir()
            if image.is_file() and is_image(image)
        ]

        if len(images) == 0:
            continue

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)

        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        copy_images(
            train_images,
            SPLIT_DATA_DIR / "train" / class_folder.name,
        )

        copy_images(
            val_images,
            SPLIT_DATA_DIR / "val" / class_folder.name,
        )

        copy_images(
            test_images,
            SPLIT_DATA_DIR / "test" / class_folder.name,
        )

        metadata[class_folder.name] = {
            "train": len(train_images),
            "validation": len(val_images),
            "test": len(test_images),
            "total": total,
        }

        total_images += total

        print(
            f"{class_folder.name:<40}"
            f" Train:{len(train_images):>5}"
            f"  Val:{len(val_images):>5}"
            f"  Test:{len(test_images):>5}"
        )

    # ------------------------------------------------------
    # Save Metadata
    # ------------------------------------------------------

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "total_classes": len(metadata),
        "total_images": total_images,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "classes": metadata,
    }

    metadata_path = METADATA_DIR / "dataset_info.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(dataset_info, file, indent=4)

    print("\n" + "=" * 60)
    print("Dataset splitting completed successfully.")
    print("=" * 60)
    print(f"Total Classes : {len(metadata)}")
    print(f"Total Images  : {total_images}")
    print(f"Metadata File : {metadata_path}")
    print(f"Output Folder : {SPLIT_DATA_DIR}")
    print("=" * 60)


# ==========================================================
# Run Directly
# ==========================================================

if __name__ == "__main__":
    split_dataset()