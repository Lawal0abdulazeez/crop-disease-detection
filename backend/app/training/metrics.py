"""
Metrics

Reusable metrics for training, validation
and testing.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ClassificationMetrics:
    """
    Collect predictions during an epoch
    and compute classification metrics.
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.targets = []

        self.predictions = []

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):

        preds = torch.argmax(outputs, dim=1)

        self.predictions.extend(
            preds.detach().cpu().numpy()
        )

        self.targets.extend(
            labels.detach().cpu().numpy()
        )

    def compute(self):

        accuracy = accuracy_score(
            self.targets,
            self.predictions,
        )

        precision = precision_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def confusion_matrix(self):

        return confusion_matrix(
            self.targets,
            self.predictions,
        )

    def classification_report(
        self,
        class_names=None,
    ):

        return classification_report(
            self.targets,
            self.predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )


# ==========================================================
# Running Average
# ==========================================================

class AverageMeter:
    """
    Computes running averages.

    Used for tracking loss during epochs.
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.value = 0.0

        self.average = 0.0

        self.total = 0.0

        self.count = 0

    def update(
        self,
        value: float,
        n: int = 1,
    ):

        self.value = value

        self.total += value * n

        self.count += n

        self.average = self.total / self.count


# ==========================================================
# Top-k Accuracy
# ==========================================================

def top_k_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    k: int = 3,
):

    _, pred = outputs.topk(
        k,
        dim=1,
    )

    correct = pred.eq(
        labels.view(-1, 1)
    )

    return correct.any(
        dim=1
    ).float().mean().item()


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    metrics = ClassificationMetrics()

    outputs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.9, 0.1],
        ]
    )

    labels = torch.tensor(
        [
            1,
            0,
            1,
            0,
        ]
    )

    metrics.update(
        outputs,
        labels,
    )

    print(metrics.compute())

    print()

    print(metrics.confusion_matrix())

    print()

    print(
        metrics.classification_report(
            ["Healthy", "Diseased"]
        )
    )