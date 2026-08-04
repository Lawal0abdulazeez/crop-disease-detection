"""
Model Evaluator

Runs evaluation on the test set and writes reports/plots.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm

from app.core.config import (
    EVAL_DIR,
    TOP_K_EVAL,
    CONFUSION_MATRIX,
    CLASSIFICATION_REPORT,
    SAVE_PREDICTIONS,
)
from app.training.metrics import ClassificationMetrics, top_k_accuracy
from app.utils.device import get_device
from app.utils.logger import get_logger

logger = get_logger()


class ModelEvaluator:
    """
    Evaluate a trained model on a DataLoader (typically test).
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        class_names: list[str],
        device=None,
        criterion: nn.Module | None = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.class_names = class_names
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.metrics = ClassificationMetrics()

        self.all_probs: list[np.ndarray] = []
        self.all_preds: list[int] = []
        self.all_labels: list[int] = []
        self.top_k_scores: list[float] = []

    @torch.no_grad()
    def run(self) -> dict:
        """
        Run full evaluation.

        Returns
        -------
        dict
            Metrics and paths to saved artifacts.
        """
        logger.info("Starting evaluation on test set...")

        loss_total = 0.0
        n_samples = 0

        progress = tqdm(
            self.data_loader,
            desc="Evaluating",
            leave=False,
        )

        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            n_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            self.metrics.update(outputs, labels)

            self.all_probs.extend(probs.cpu().numpy())
            self.all_preds.extend(preds.cpu().numpy().tolist())
            self.all_labels.extend(labels.cpu().numpy().tolist())

            k = min(TOP_K_EVAL, outputs.size(1))
            self.top_k_scores.append(
                top_k_accuracy(outputs, labels, k=k)
            )

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = loss_total / max(n_samples, 1)
        metric_dict = self.metrics.compute()
        top_k_mean = (
            float(np.mean(self.top_k_scores)) if self.top_k_scores else 0.0
        )

        results = {
            "loss": avg_loss,
            "accuracy": metric_dict["accuracy"],
            "precision": metric_dict["precision"],
            "recall": metric_dict["recall"],
            "f1": metric_dict["f1"],
            f"top_{TOP_K_EVAL}_accuracy": top_k_mean,
            "num_samples": n_samples,
            "num_classes": len(self.class_names),
        }

        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Samples          : {n_samples}")
        logger.info(f"Loss             : {avg_loss:.4f}")
        logger.info(f"Accuracy         : {metric_dict['accuracy']:.4f}")
        logger.info(f"Precision        : {metric_dict['precision']:.4f}")
        logger.info(f"Recall           : {metric_dict['recall']:.4f}")
        logger.info(f"F1 Score         : {metric_dict['f1']:.4f}")
        logger.info(f"Top-{TOP_K_EVAL} Accuracy  : {top_k_mean:.4f}")
        logger.info("=" * 70)

        EVAL_DIR.mkdir(parents=True, exist_ok=True)

        metrics_path = EVAL_DIR / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        if CLASSIFICATION_REPORT:
            report = self.metrics.classification_report(
                class_names=self.class_names
            )
            report_path = EVAL_DIR / "classification_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Classification report saved to {report_path}")
            print("\n" + report)

        if CONFUSION_MATRIX:
            cm = self.metrics.confusion_matrix()
            self._plot_confusion_matrix(cm)

        if SAVE_PREDICTIONS:
            preds_path = EVAL_DIR / "predictions.json"
            payload = {
                "labels": self.all_labels,
                "predictions": self.all_preds,
                "class_names": self.class_names,
            }
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            logger.info(f"Predictions saved to {preds_path}")

        results["artifacts"] = {
            "metrics": str(metrics_path),
            "evaluation_dir": str(EVAL_DIR),
        }
        return results

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = EVAL_DIR / "confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved to {path}")
