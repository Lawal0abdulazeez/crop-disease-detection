"""
Logging Utilities

Centralized logging configuration for the Crop Disease Detection project.

Logs are written to both the console and a log file.

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import LOG_DIR

# ==========================================================
# Log Configuration
# ==========================================================

LOG_FILE = LOG_DIR / "training.log"

LOGGER_NAME = "crop_disease_detection"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==========================================================
# Logger Factory
# ==========================================================

def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """
    Create and return a configured logger.

    Multiple calls return the same logger without
    duplicating handlers.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # ----------------------------------------------
    # Console Handler
    # ----------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    console_handler.setLevel(logging.INFO)

    # ----------------------------------------------
    # File Handler
    # ----------------------------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    file_handler.setLevel(logging.INFO)

    # ----------------------------------------------

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# ==========================================================
# Helper Functions
# ==========================================================

def log_separator(logger: logging.Logger, length: int = 70) -> None:
    """
    Log a separator line.
    """

    logger.info("=" * length)


def log_title(logger: logging.Logger, title: str) -> None:
    """
    Log a formatted title.
    """

    logger.info("=" * 70)

    logger.info(title)

    logger.info("=" * 70)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    logger = get_logger()

    log_title(logger, "LOGGER TEST")

    logger.info("Information message")

    logger.warning("Warning message")

    logger.error("Error message")

    logger.info("Logger initialized successfully.")