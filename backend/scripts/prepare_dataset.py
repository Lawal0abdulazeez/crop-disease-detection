"""
Dataset Preparation Script

This script prepares the PlantVillage dataset
for model training.

Pipeline
--------
1. Initialize project folders
2. Verify raw dataset
3. Split dataset
4. Create metadata
5. Validate datasets
6. Validate dataloaders

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from pathlib import Path

from app.core.config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
)

from app.core.paths import initialize_project

from app.data.splitter import split_dataset

from app.data.dataset import (
    get_dataset_summary,
)

from app.data.dataloader import (
    create_dataloaders,
)


# ==========================================================
# Verify Raw Dataset
# ==========================================================

def verify_raw_dataset() -> None:
    """
    Verify that a dataset exists inside data/raw.
    """

    if not RAW_DATA_DIR.exists():

        raise FileNotFoundError(
            f"\nRaw dataset directory does not exist:\n{RAW_DATA_DIR}"
        )

    dataset_folders = [
        folder
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir()
    ]

    if len(dataset_folders) == 0:

        raise FileNotFoundError(
            "\nNo dataset found inside data/raw.\n"
            "Download the dataset before running this script."
        )

    print("✓ Raw dataset found.")


# ==========================================================
# Verify Split
# ==========================================================

def verify_split() -> None:
    """
    Verify train/val/test folders exist.
    """

    required = [
        SPLIT_DATA_DIR / "train",
        SPLIT_DATA_DIR / "val",
        SPLIT_DATA_DIR / "test",
    ]

    for folder in required:

        if not folder.exists():

            raise FileNotFoundError(
                f"Missing dataset split:\n{folder}"
            )

    print("✓ Dataset split verified.")


# ==========================================================
# Print Summary
# ==========================================================

def print_summary() -> None:

    summary = get_dataset_summary()

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Classes              : {summary['num_classes']}")

    print(f"Training Images      : {summary['train_samples']}")

    print(f"Validation Images    : {summary['validation_samples']}")

    print(f"Testing Images       : {summary['test_samples']}")

    print("=" * 70)


# ==========================================================
# Verify DataLoader
# ==========================================================

def verify_dataloader() -> None:

    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("\n")
    print("=" * 70)
    print("DATALOADER SUMMARY")
    print("=" * 70)

    print(f"Train Batches        : {len(train_loader)}")

    print(f"Validation Batches   : {len(val_loader)}")

    print(f"Testing Batches      : {len(test_loader)}")

    print()

    print(f"Image Tensor Shape   : {tuple(images.shape)}")

    print(f"Label Tensor Shape   : {tuple(labels.shape)}")

    print("=" * 70)


# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    print("\n")
    print("=" * 70)
    print("PLANTVILLAGE DATA PREPARATION")
    print("=" * 70)

    print("\nStep 1: Initializing Project Directories...")
    initialize_project(verbose=False)
    print("✓ Directories initialized.")

    print("\nStep 2: Checking Raw Dataset...")
    verify_raw_dataset()

    print("\nStep 3: Splitting Dataset...")
    split_dataset()

    print("\nStep 4: Verifying Dataset Split...")
    verify_split()

    print("\nStep 5: Creating Dataset Summary...")
    print_summary()

    print("\nStep 6: Validating DataLoaders...")
    verify_dataloader()

    print("\n")
    print("=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()