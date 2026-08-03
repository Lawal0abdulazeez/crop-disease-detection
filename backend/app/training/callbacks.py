"""
Training Callbacks

Contains reusable callbacks for model training.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations


# ==========================================================
# Early Stopping
# ==========================================================

class EarlyStopping:
    """
    Stop training if validation metric stops improving.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None

        self.counter = 0

        self.should_stop = False

    def update(
        self,
        value: float,
    ) -> bool:

        if self.best_score is None:

            self.best_score = value

            return False

        if self.mode == "min":

            improved = value < (
                self.best_score - self.min_delta
            )

        else:

            improved = value > (
                self.best_score + self.min_delta
            )

        if improved:

            self.best_score = value

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop