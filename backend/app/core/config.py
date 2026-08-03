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



# ==========================================================
# Dataset
# ==========================================================

IMAGE_SIZE = 224

NUM_CHANNELS = 3

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# ==========================================================
# Data Split
# ==========================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==========================================================
# DataLoader
# ==========================================================

BATCH_SIZE = 32

NUM_WORKERS = 4

PIN_MEMORY = True

SHUFFLE_TRAIN = True

# ==========================================================
# Training Modes
# ==========================================================

TRAIN_MODE = "debug"

TRAIN_MODES = {

    "smoke": {
        "train_samples": 32,
        "val_samples": 16,
        "test_samples": 16,
        "epochs": 1,
    },

    "debug": {
        "train_samples": 500,
        "val_samples": 100,
        "test_samples": 100,
        "epochs": 2,
    },

    "prototype": {
        "train_samples": 2000,
        "val_samples": 400,
        "test_samples": 400,
        "epochs": 5,
    },

    "full": {
        "train_samples": None,
        "val_samples": None,
        "test_samples": None,
        "epochs": 25,
    },
}

# ==========================================================
# Model
# ==========================================================

MODEL_NAME = "efficientnet_b0"

PRETRAINED = True

FREEZE_BACKBONE = False

DROPOUT = 0.30

# ==========================================================
# Optimizer
# ==========================================================

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

OPTIMIZER = "adamw"

# ==========================================================
# Scheduler
# ==========================================================

SCHEDULER = "reduce_on_plateau"

LR_PATIENCE = 3

LR_FACTOR = 0.1

MIN_LEARNING_RATE = 1e-6

# ==========================================================
# Training
# ==========================================================

LOSS_FUNCTION = "cross_entropy"

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 5

SAVE_BEST_ONLY = True

SAVE_EVERY_EPOCH = False

GRADIENT_CLIP = 1.0

MIXED_PRECISION = True

# ==========================================================
# Logging
# ==========================================================

LOG_INTERVAL = 10

VERBOSE = True

# ==========================================================
# Evaluation
# ==========================================================

CONFUSION_MATRIX = True

CLASSIFICATION_REPORT = True

SAVE_PREDICTIONS = True

# ==========================================================
# Checkpoints
# ==========================================================

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pt"

SAVE_EVERY_N_EPOCHS = 5

RESUME_TRAINING = False

RESUME_CHECKPOINT = None

# ==========================================================
# Class Imbalance
# ==========================================================

USE_CLASS_WEIGHTS = False

CLASS_WEIGHTS = None

LABEL_SMOOTHING = 0.0


# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 7

EARLY_STOPPING_MIN_DELTA = 0.0001

MONITOR_METRIC = "val_loss"

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

SCHEDULER = "reduce_on_plateau"
# Options:
#   reduce_on_plateau
#   cosine
#   step
#   exponential
#   none

STEP_SIZE = 5

STEP_GAMMA = 0.1

COSINE_T_MAX = NUM_EPOCHS

ETA_MIN = 1e-6

REDUCE_FACTOR = 0.5

REDUCE_PATIENCE = 2

EXPONENTIAL_GAMMA = 0.95

# ==========================================================
# Optimizer
# ==========================================================

OPTIMIZER = "adam"

# Supported:
# adam
# adamw
# sgd
# rmsprop

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

MOMENTUM = 0.9

NESTEROV = True

# ==========================================================
# Training Modes
# ==========================================================

TRAINING_MODE = "full"

# Options:
# debug
# smoke
# full

MAX_TRAIN_BATCHES = None

MAX_VAL_BATCHES = None


# ==========================================================
# Training Modes
# ==========================================================

TRAINING_MODE = "full"

# full
# debug
# smoke

MAX_TRAIN_BATCHES = None

MAX_VAL_BATCHES = None

# ==========================================================
# Checkpoint Configuration
# ==========================================================

SAVE_LAST = True

SAVE_BEST = True

SAVE_EVERY_N_EPOCHS = 5

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

BEST_MODEL_NAME = "best_model.pt"

LAST_MODEL_NAME = "last_model.pt"

# ==========================================================
# Training History
# ==========================================================

HISTORY_DIR = OUTPUTS_DIR / "history"

PLOTS_DIR = OUTPUTS_DIR / "plots"

SAVE_HISTORY = True

SAVE_PLOTS = True