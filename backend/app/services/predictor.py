"""
Prediction Service

Loads a checkpoint once and serves single or batch predictions.
Shared by the CLI predict script and the FastAPI layer.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.core.config import (
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LAST_MODEL_NAME,
    MODEL_NAME,
    CLASS_NAMES_FILE,
    TOP_K_PREDICTIONS,
    CONFIDENCE_THRESHOLD,
)
from app.data.transforms import predict_transform
from app.models.factory import create_model
from app.models.checkpoint import load_checkpoint
from app.utils.device import get_device
from app.utils.logger import get_logger

logger = get_logger()


class PredictionService:
    """
    Singleton-style predictor. Call load() once, then predict().
    """

    def __init__(self):
        self.model = None
        self.device = get_device()
        self.class_names: list[str] = []
        self.model_name: str = MODEL_NAME
        self.num_classes: int = 0
        self.checkpoint_path: Path | None = None
        self._loaded = False

    def load(self, checkpoint: str | Path = "best") -> None:
        """
        Load model weights and class names.

        Parameters
        ----------
        checkpoint
            'best', 'last', or path to a .pt file.
        """
        if isinstance(checkpoint, str):
            if checkpoint.lower() == "best":
                path = CHECKPOINT_DIR / BEST_MODEL_NAME
            elif checkpoint.lower() == "last":
                path = CHECKPOINT_DIR / LAST_MODEL_NAME
            else:
                path = Path(checkpoint)
        else:
            path = Path(checkpoint)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Train first: uv run python -m scripts.train"
            )

        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.num_classes = int(raw.get("num_classes", 0))
        self.model_name = raw.get("model_name", MODEL_NAME)

        if self.num_classes <= 0:
            if CLASS_NAMES_FILE.exists():
                with open(CLASS_NAMES_FILE, encoding="utf-8") as f:
                    names = json.load(f)
                self.num_classes = len(names)
            else:
                raise ValueError(
                    "Cannot determine num_classes from checkpoint or class_names.json"
                )

        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            show_summary=False,
        )
        load_checkpoint(checkpoint_path=path, model=self.model, device=self.device)
        self.model.eval()

        if CLASS_NAMES_FILE.exists():
            with open(CLASS_NAMES_FILE, encoding="utf-8") as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

        self.checkpoint_path = path
        self._loaded = True
        logger.info(
            f"PredictionService loaded: {path.name} "
            f"({self.num_classes} classes, device={self.device})"
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load("best")

    def _prepare_image(self, image: Image.Image | Path | str) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = predict_transform(image)
        return tensor.unsqueeze(0)  # (1, C, H, W)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | Path | str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Predict disease for a single image.

        Returns
        -------
        dict with predicted_class, confidence, top_k list, etc.
        """
        self._ensure_loaded()
        top_k = top_k or TOP_K_PREDICTIONS
        top_k = min(top_k, self.num_classes)

        tensor = self._prepare_image(image).to(self.device)
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze(0)

        values, indices = torch.topk(probs, k=top_k)

        top_predictions = [
            {
                "class_index": int(idx),
                "class_name": self.class_names[int(idx)],
                "confidence": float(val),
            }
            for val, idx in zip(values.tolist(), indices.tolist())
        ]

        best = top_predictions[0]

        return {
            "predicted_class": best["class_name"],
            "class_index": best["class_index"],
            "confidence": best["confidence"],
            "above_threshold": best["confidence"] >= CONFIDENCE_THRESHOLD,
            "top_k": top_predictions,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
        }

    def predict_batch(
        self,
        images: list[Image.Image | Path | str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Predict for a list of images."""
        return [self.predict(img, top_k=top_k) for img in images]

    def info(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": str(self.device),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "top_k": TOP_K_PREDICTIONS,
        }


# Module-level shared instance (lazy-loaded on first use)
predictor = PredictionService()
