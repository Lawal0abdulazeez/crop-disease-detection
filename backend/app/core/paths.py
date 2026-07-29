"""
Project Path Utilities

This module ensures that all required project directories
exist before any training, preprocessing or inference begins.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from pathlib import Path

from app.core.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DATA_DIR,
    METADATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    EXPORT_DIR,
    LOG_DIR,
)

# ==========================================================
# Directories Required By The Project
# ==========================================================

REQUIRED_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DATA_DIR,
    METADATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    EXPORT_DIR,
    LOG_DIR,
]


def create_directories(verbose: bool = True) -> None:
    """
    Create all project directories if they do not already exist.

    Args:
        verbose (bool):
            Whether to print created directories.
    """

    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"✓ {directory}")


def ensure_directory(path: Path) -> Path:
    """
    Ensure a single directory exists.

    Args:
        path (Path): Directory path.

    Returns:
        Path: The created (or existing) directory.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize_project(verbose: bool = True) -> None:
    """
    Initialize the project's directory structure.
    Safe to call multiple times.

    Args:
        verbose (bool):
            Whether to display created directories.
    """

    if verbose:
        print("=" * 60)
        print("Initializing Project Directory Structure")
        print("=" * 60)

    create_directories(verbose=verbose)

    if verbose:
        print("=" * 60)
        print("Project directories are ready.")
        print("=" * 60)


# ==========================================================
# Run Directly
# ==========================================================

if __name__ == "__main__":
    initialize_project()