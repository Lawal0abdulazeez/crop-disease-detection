"""
Model Training Script

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import torch

from app.utils.seed import seed_everything
from app.utils.logger import get_logger

from app.data.dataloader import (
    create_dataloaders,
)

from app.models.factory import (
    create_model,
)

from app.training.optimizer import (
    create_optimizer,
)

from app.training.scheduler import (
    create_scheduler,
)

from app.training.losses import (
    create_loss,
)

from app.training.trainer import (
    Trainer,
)

from app.core.config import (
    RANDOM_SEED,
    MODEL_NAME,
    NUM_CLASSES,
)

logger = get_logger()


def main():

    logger.info("=" * 70)
    logger.info("Crop Disease Detection Training")
    logger.info("=" * 70)

    # ------------------------------------
    # Reproducibility
    # ------------------------------------

    seed_everything(RANDOM_SEED)

    # ------------------------------------
    # Dataloaders
    # ------------------------------------

    logger.info("Creating dataloaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    logger.info("Done.")

    # ------------------------------------
    # Model
    # ------------------------------------

    logger.info("Creating model...")

    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
    )

    # ------------------------------------
    # Optimizer
    # ------------------------------------

    optimizer = create_optimizer(
        model
    )

    # ------------------------------------
    # Scheduler
    # ------------------------------------

    scheduler = create_scheduler(
        optimizer
    )

    # ------------------------------------
    # Loss
    # ------------------------------------

    criterion = create_loss()

    # ------------------------------------
    # Trainer
    # ------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    history = trainer.train(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        resume=False,
    )

    logger.info("Training Complete.")

    return history


if __name__ == "__main__":

    main()