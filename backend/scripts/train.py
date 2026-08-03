"""
Model Training Script

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

from app.utils.seed import seed_everything
from app.utils.logger import get_logger

from app.data.dataloader import create_dataloaders
from app.data.dataset import get_num_classes

from app.models.factory import create_model

from app.training.optimizer import create_optimizer
from app.training.scheduler import create_scheduler
from app.training.losses import create_loss
from app.training.trainer import Trainer

from app.core.config import (
    RANDOM_SEED,
    MODEL_NAME,
)
from app.core.paths import initialize_project

logger = get_logger()


def main():
    logger.info("=" * 70)
    logger.info("Crop Disease Detection Training")
    logger.info("=" * 70)

    # Ensure directories exist
    initialize_project(verbose=False)

    # Reproducibility
    seed_everything(RANDOM_SEED)

    # Dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders()
    logger.info("Done.")

    # Number of classes from the actual dataset
    num_classes = get_num_classes()
    logger.info(f"Number of classes: {num_classes}")

    # Model
    logger.info("Creating model...")
    model = create_model(
        model_name=MODEL_NAME,
        num_classes=num_classes,
    )

    # Optimizer / Scheduler / Loss
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    criterion = create_loss()

    # Trainer
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
        num_classes=num_classes,
        resume=False,
    )

    logger.info("Training Complete.")
    return history


if __name__ == "__main__":
    main()
