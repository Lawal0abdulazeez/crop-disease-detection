"""
CLI Prediction Script

Usage
-----
  uv run python -m scripts.predict path/to/leaf.jpg
  uv run python -m scripts.predict path/to/leaf.jpg --checkpoint last --top-k 5

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.predictor import PredictionService
from app.utils.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from leaf image")
    parser.add_argument("image", type=str, help="Path to leaf image")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'last', or path to .pt (default: best)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    service = PredictionService()
    service.load(args.checkpoint)

    result = service.predict(image_path, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"Image            : {image_path}")
    print(f"Predicted class  : {result['predicted_class']}")
    print(f"Confidence       : {result['confidence']:.4f}")
    print(f"Above threshold  : {result['above_threshold']}")
    print("-" * 60)
    print("Top-K:")
    for i, item in enumerate(result["top_k"], 1):
        print(
            f"  {i}. {item['class_name']:<40} "
            f"{item['confidence']:.4f}"
        )
    print("=" * 60)

    # Also print JSON for scripting
    print("\nJSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
