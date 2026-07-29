"""
Project Configuration

This module contains all configurable settings used across the project.

It centralizes:
- Project paths
- Dataset paths
- Model paths
- Training hyperparameters
- Image settings
- Dataset split ratios

Author: Abdulazeez Lawal
Project: Crop Disease Detection
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

# ============================================================
# Project Root
# ============================================================

# backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# Dataset Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SPLIT_DATA_DIR = DATA_DIR / "splits"

METADATA_DIR = DATA_DIR / "metadata"

# ============================================================
# Model Directories
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

EXPORT_DIR = MODEL_DIR / "exports"

# ============================================================
# Logging
# ============================================================

LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# Training Configuration
# ============================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 4

PIN_MEMORY = True

SHUFFLE_TRAIN = True

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_EPOCHS = 20

EARLY_STOPPING_PATIENCE = 5

DEVICE = "cuda"

# ============================================================
# Dataset Split Ratios
# ============================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

# Ensure ratios sum to 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# ============================================================
# Image Formats
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

# ============================================================
# Saved Model Names
# ============================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"

# ============================================================
# Environment Variables
# ============================================================

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")

KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# ============================================================
# API Configuration (For FastAPI Later)
# ============================================================

API_HOST = "0.0.0.0"

API_PORT = 8000

# ============================================================
# Prediction Configuration
# ============================================================

TOP_K_PREDICTIONS = 3

CONFIDENCE_THRESHOLD = 0.50

# ============================================================
# Miscellaneous
# ============================================================

CLASS_NAMES_FILE = METADATA_DIR / "class_names.json"

DATASET_INFO_FILE = METADATA_DIR / "dataset_info.json"

TRAINING_HISTORY_FILE = METADATA_DIR / "training_history.json"