"""
Dataset Split Tests

Verifies that the dataset splitting pipeline
produced a valid train/validation/test structure.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from pathlib import Path

from app.core.config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ==========================================================
# Helper Functions
# ==========================================================

def get_dataset_root() -> Path:
    """
    Returns the dataset folder inside data/raw.
    """

    folders = [
        folder
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir()
    ]

    if not folders:
        raise FileNotFoundError(
            "No dataset found inside data/raw."
        )

    return folders[0]


def count_images(folder: Path) -> int:
    """
    Counts all supported images recursively.
    """

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


# ==========================================================
# Tests
# ==========================================================

def test_split_directories_exist():

    assert SPLIT_DATA_DIR.exists()

    assert (SPLIT_DATA_DIR / "train").exists()

    assert (SPLIT_DATA_DIR / "val").exists()

    assert (SPLIT_DATA_DIR / "test").exists()


def test_class_folders_exist():

    raw_root = get_dataset_root()

    raw_classes = sorted(
        [
            folder.name
            for folder in raw_root.iterdir()
            if folder.is_dir()
        ]
    )

    train_classes = sorted(
        [
            folder.name
            for folder in (SPLIT_DATA_DIR / "train").iterdir()
            if folder.is_dir()
        ]
    )

    val_classes = sorted(
        [
            folder.name
            for folder in (SPLIT_DATA_DIR / "val").iterdir()
            if folder.is_dir()
        ]
    )

    test_classes = sorted(
        [
            folder.name
            for folder in (SPLIT_DATA_DIR / "test").iterdir()
            if folder.is_dir()
        ]
    )

    assert raw_classes == train_classes
    assert raw_classes == val_classes
    assert raw_classes == test_classes


def test_no_empty_class_folder():

    for split in ["train", "val", "test"]:

        split_dir = SPLIT_DATA_DIR / split

        for class_folder in split_dir.iterdir():

            if class_folder.is_dir():

                images = count_images(class_folder)

                assert images > 0, (
                    f"{split}/{class_folder.name} "
                    "contains no images."
                )


def test_total_images_preserved():

    raw_root = get_dataset_root()

    raw_images = count_images(raw_root)

    train_images = count_images(
        SPLIT_DATA_DIR / "train"
    )

    val_images = count_images(
        SPLIT_DATA_DIR / "val"
    )

    test_images = count_images(
        SPLIT_DATA_DIR / "test"
    )

    split_total = (
        train_images +
        val_images +
        test_images
    )

    assert raw_images == split_total


def test_each_class_contains_images():

    for split in ["train", "val", "test"]:

        split_dir = SPLIT_DATA_DIR / split

        for class_folder in split_dir.iterdir():

            if class_folder.is_dir():

                images = list(
                    class_folder.glob("*")
                )

                assert len(images) > 0


def test_dataset_not_empty():

    train = count_images(
        SPLIT_DATA_DIR / "train"
    )

    val = count_images(
        SPLIT_DATA_DIR / "val"
    )

    test = count_images(
        SPLIT_DATA_DIR / "test"
    )

    assert train > 0

    assert val > 0

    assert test > 0