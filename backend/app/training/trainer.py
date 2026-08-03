"""
Training Engine

Handles model training, validation,
checkpointing and logging.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""
import json

import matplotlib.pyplot as plt

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from tqdm import tqdm

from app.training.metrics import (
    ClassificationMetrics,
    AverageMeter,
)

from app.models.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


from app.training.callbacks import (
    EarlyStopping,
    MetricTracker,
)

from app.training.scheduler import (
    scheduler_requires_metric,
)



from app.utils.device import (
    get_device,
)

from app.utils.logger import (
    get_logger,
)

from app.core.config import (
    NUM_EPOCHS,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    CHECKPOINT_DIR,
    LAST_MODEL_NAME,
    BEST_MODEL_NAME,
    HISTORY_DIR,
    PLOTS_DIR,
    SAVE_HISTORY,
    SAVE_PLOTS,
)

logger = get_logger()


class Trainer:
    """
    Generic Trainer
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device=None,
    ):

        self.device = device or get_device()

        self.model = model.to(self.device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        self.scaler = GradScaler(
            enabled=self.device.type == "cuda"
        )

        self.metric_tracker = MetricTracker()

        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            mode="min",
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        logger.info(
            "Trainer initialized successfully."
        )

        logger.info(
            f"Training batches : {len(train_loader)}"
        )

        logger.info(
            f"Validation batches : {len(val_loader)}"
        )

        logger.info(
            f"Device : {self.device}"
        )

        # ==========================================================
    # Train One Epoch
    # ==========================================================

    def train_one_epoch(
        self,
        epoch: int,
    ) -> dict:
        """
        Train the model for one epoch.

        Returns
        -------
        dict
            Training metrics.
        """

        self.model.train()

        metrics = ClassificationMetrics()

        loss_meter = AverageMeter()

        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        for batch_idx, (images, labels) in progress:

            # ------------------------------------------
            # Debug / Smoke mode
            # ------------------------------------------

            if (
                self.max_train_batches is not None
                and batch_idx >= self.max_train_batches
            ):
                break

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

            # ------------------------------------------
            # Backward
            # ------------------------------------------

            self.scaler.scale(loss).backward()

            # ------------------------------------------
            # Gradient Clipping
            # ------------------------------------------

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            batch_size = labels.size(0)

            loss_meter.update(
                loss.item(),
                batch_size,
            )

            metrics.update(
                outputs,
                labels,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            progress.set_postfix(
                loss=f"{loss_meter.average:.4f}",
                lr=f"{current_lr:.6f}",
            )

        epoch_metrics = metrics.compute()

        epoch_metrics["loss"] = loss_meter.average

        epoch_metrics["learning_rate"] = current_lr

        logger.info(
            (
                f"[Train] "
                f"Epoch {epoch:03d} | "
                f"Loss={loss_meter.average:.4f} | "
                f"Acc={epoch_metrics['accuracy']:.4f}"
            )
        )

        return epoch_metrics

        # ==========================================================
    # Validate One Epoch
    # ==========================================================

    def validate_one_epoch(
        self,
        epoch: int,
    ) -> dict:
        """
        Validate the model for one epoch.

        Returns
        -------
        dict
            Validation metrics.
        """

        self.model.eval()

        metrics = ClassificationMetrics()

        loss_meter = AverageMeter()

        progress = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"Validation Epoch {epoch}",
            leave=False,
        )

        with torch.no_grad():

            for batch_idx, (images, labels) in progress:

                # ------------------------------------------
                # Debug / Smoke mode
                # ------------------------------------------

                if (
                    self.max_val_batches is not None
                    and batch_idx >= self.max_val_batches
                ):
                    break

                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True,
                )

                # ------------------------------------------
                # Forward
                # ------------------------------------------

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda",
                ):

                    outputs = self.model(images)

                    loss = self.criterion(
                        outputs,
                        labels,
                    )

                # ------------------------------------------
                # Statistics
                # ------------------------------------------

                batch_size = labels.size(0)

                loss_meter.update(
                    loss.item(),
                    batch_size,
                )

                metrics.update(
                    outputs,
                    labels,
                )

                progress.set_postfix(
                    loss=f"{loss_meter.average:.4f}",
                )

        epoch_metrics = metrics.compute()

        epoch_metrics["loss"] = loss_meter.average

        logger.info(
            (
                f"[Validation] "
                f"Epoch {epoch:03d} | "
                f"Loss={loss_meter.average:.4f} | "
                f"Acc={epoch_metrics['accuracy']:.4f}"
            )
        )

        return epoch_metrics
    

        # ==========================================================
    # Training Loop
    # ==========================================================

        
    def train(
        self,
        num_epochs=NUM_EPOCHS,
        model_name="efficientnet_b0",
        num_classes=15,
        resume=False,
    ):
        """
        Execute the full training loop.

        Returns
        -------
        dict
            Training history.
        """

        logger.info("=" * 70)
        logger.info("TRAINING STARTED")
        logger.info("=" * 70)


        start_epoch = 1

        if resume:

            start_epoch = self.resume_checkpoint()

        for epoch in range(
            start_epoch,
            num_epochs + 1,
        ):

            logger.info(
                f"Epoch {epoch}/{num_epochs}"
            )

            # --------------------------------------------------
            # Train
            # --------------------------------------------------

            train_metrics = self.train_one_epoch(epoch)

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            val_metrics = self.validate_one_epoch(epoch)

            # --------------------------------------------------
            # Scheduler
            # --------------------------------------------------

            if self.scheduler is not None:

                if scheduler_requires_metric(self.scheduler):

                    self.scheduler.step(
                        val_metrics["loss"]
                    )

                else:

                    self.scheduler.step()

            # --------------------------------------------------
            # Update History
            # --------------------------------------------------

            self.history["train_loss"].append(
                train_metrics["loss"]
            )

            self.history["val_loss"].append(
                val_metrics["loss"]
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            self.history["train_precision"].append(
                train_metrics["precision"]
            )

            self.history["val_precision"].append(
                val_metrics["precision"]
            )

            self.history["train_recall"].append(
                train_metrics["recall"]
            )

            self.history["val_recall"].append(
                val_metrics["recall"]
            )

            self.history["train_f1"].append(
                train_metrics["f1"]
            )

            self.history["val_f1"].append(
                val_metrics["f1"]
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["learning_rate"].append(
                current_lr
            )

            # --------------------------------------------------
            # Best Metrics
            # --------------------------------------------------

            self.metric_tracker.update(
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
            )

            is_best = (
                self.metric_tracker.is_best_accuracy
            )

            # --------------------------------------------------
            # Save Checkpoint
            # --------------------------------------------------

            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"],
                model_name=model_name,
                num_classes=num_classes,
                is_best=is_best,
            )

            # --------------------------------------------------
            # Epoch Summary
            # --------------------------------------------------

            logger.info(
                "-" * 70
            )

            logger.info(
                f"Train Loss : {train_metrics['loss']:.4f}"
            )

            logger.info(
                f"Val Loss   : {val_metrics['loss']:.4f}"
            )

            logger.info(
                f"Train Acc  : {train_metrics['accuracy']:.4f}"
            )

            logger.info(
                f"Val Acc    : {val_metrics['accuracy']:.4f}"
            )

            logger.info(
                f"Learning Rate : {current_lr:.6f}"
            )

            logger.info(
                "-" * 70
            )

            # --------------------------------------------------
            # Early Stopping
            # --------------------------------------------------

            if EARLY_STOPPING:

                stop = self.early_stopping.update(
                    val_metrics["loss"]
                )

                if stop:

                    logger.info(
                        "Early stopping triggered."
                    )

                    break

        logger.info("=" * 70)

        logger.info("TRAINING FINISHED")

        logger.info("=" * 70)

        return self.history

        # ==========================================================
    # Resume Training
    # ==========================================================

    def resume_checkpoint(self):
        """
        Resume training from the latest checkpoint.

        Returns
        -------
        int
            Epoch to continue from.
        """

        checkpoint_path = (
            CHECKPOINT_DIR / LAST_MODEL_NAME
        )

        if not checkpoint_path.exists():

            logger.info(
                "No previous checkpoint found."
            )

            return 1

        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        start_epoch = checkpoint["epoch"] + 1

        logger.info(
            f"Resumed from epoch {checkpoint['epoch']}"
        )

        logger.info(
            f"Validation Accuracy: "
            f"{checkpoint['val_accuracy']:.4f}"
        )

        return start_epoch

    # ==========================================================
    # Save History
    # ==========================================================

    def save_history(self):
        """
        Save training history as JSON.
        """

        if not SAVE_HISTORY:
            return

        HISTORY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            HISTORY_DIR /
            "training_history.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.history,
                f,
                indent=4,
            )

        logger.info(
            f"Training history saved to {output_file}"
        )


        # ==========================================================
    # Plot History
    # ==========================================================

    def plot_history(self):
        """
        Generate training plots.
        """

        if not SAVE_PLOTS:
            return

        PLOTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        epochs = range(
            1,
            len(self.history["train_loss"]) + 1,
        )

        # -------------------------------
        # Loss
        # -------------------------------

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            self.history["train_loss"],
            label="Train",
        )

        plt.plot(
            epochs,
            self.history["val_loss"],
            label="Validation",
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Loss Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "loss_curve.png"
        )

        plt.close()

        # -------------------------------
        # Accuracy
        # -------------------------------

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            self.history["train_accuracy"],
            label="Train",
        )

        plt.plot(
            epochs,
            self.history["val_accuracy"],
            label="Validation",
        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.title("Accuracy Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "accuracy_curve.png"
        )

        plt.close()

        # -------------------------------
        # Learning Rate
        # -------------------------------

        plt.figure(figsize=(8,5))

        plt.plot(
            epochs,
            self.history["learning_rate"],
        )

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.title("Learning Rate")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            "learning_rate.png"
        )

        plt.close()

        logger.info(
            f"Training plots saved to {PLOTS_DIR}"
        )