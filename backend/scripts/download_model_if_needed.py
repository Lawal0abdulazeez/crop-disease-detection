"""
Download model checkpoint when MODEL_URL is set and local file is missing.

Used on Render / Docker boot so the image stays small.

Env:
  MODEL_URL          – direct download URL (e.g. Hugging Face / Google Drive direct link)
  MODEL_FILENAME     – default best_model.pt
  CLASS_NAMES_URL    – optional JSON of class names
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
METADATA_DIR = Path(__file__).resolve().parents[1] / "data" / "metadata"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    model_url = os.getenv("MODEL_URL", "").strip()
    filename = os.getenv("MODEL_FILENAME", "best_model.pt").strip()
    dest = CHECKPOINT_DIR / filename

    if dest.exists():
        print(f"Checkpoint already present: {dest}")
    elif model_url:
        download(model_url, dest)
    else:
        print(
            "No local checkpoint and MODEL_URL not set. "
            "API will start but /predict returns 503 until a model is available."
        )

    class_url = os.getenv("CLASS_NAMES_URL", "").strip()
    class_dest = METADATA_DIR / "class_names.json"
    if class_url and not class_dest.exists():
        download(class_url, class_dest)


if __name__ == "__main__":
    main()
