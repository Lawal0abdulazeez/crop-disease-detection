from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Dataset name
DATASET_NAME = "mohanty/PlantVillage"

# Output folder
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "plantvillage"


def save_split(split_name):
    print(f"\nDownloading {split_name} split...")

    dataset = load_dataset(
        DATASET_NAME,
        name="default",
        split=split_name,
        )
    
    #dataset = load_dataset(DATASET_NAME, split=split_name)

    print(dataset.features)

    labels = dataset.features["label"].names

    for idx, sample in enumerate(tqdm(dataset)):

        image = sample["image"]
        label = labels[sample["label"]]

        class_folder = OUTPUT_DIR / split_name / label
        class_folder.mkdir(parents=True, exist_ok=True)

        image.save(class_folder / f"{idx}.jpg")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_split("train")

    try:
        save_split("test")
    except Exception:
        print("No test split found.")


if __name__ == "__main__":
    main()