"""
Project Configuration

Centralized settings for the Crop Disease Detection project.

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
# Project Root (backend/)
# ============================================================

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
# Model & Output Directories
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
MODELS_DIR = MODEL_DIR  # alias used by some modules
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
EXPORT_DIR = MODEL_DIR / "exports"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = OUTPUTS_DIR / "history"
PLOTS_DIR = OUTPUTS_DIR / "plots"

LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42

# ============================================================
# Image Settings
# ============================================================

IMAGE_SIZE = 224
NUM_CHANNELS = 3

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# ============================================================
# Dataset Split Ratios
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

# ============================================================
# DataLoader
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True

# ============================================================
# Model
# ============================================================

MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
FREEZE_BACKBONE = False
DROPOUT = 0.30

# NUM_CLASSES is determined dynamically from the dataset at runtime.
# Do not hard-code it here.

# ============================================================
# Optimizer
# ============================================================

OPTIMIZER = "adamw"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NESTEROV = True

# ============================================================
# Scheduler
# ============================================================

SCHEDULER = "reduce_on_plateau"
# Options: none | step | cosine | reduce_on_plateau | exponential

STEP_SIZE = 5
STEP_GAMMA = 0.1

COSINE_T_MAX = 20
ETA_MIN = 1e-6

REDUCE_FACTOR = 0.5
REDUCE_PATIENCE = 2

EXPONENTIAL_GAMMA = 0.95

# ============================================================
# Loss
# ============================================================

LOSS_FUNCTION = "cross_entropy"
LABEL_SMOOTHING = 0.0
USE_CLASS_WEIGHTS = False
CLASS_WEIGHTS = None

# ============================================================
# Training
# ============================================================

NUM_EPOCHS = 20
GRADIENT_CLIP = 1.0
MIXED_PRECISION = True

# Training mode: "full" | "debug" | "smoke"
# - full  : use entire dataset, NUM_EPOCHS epochs
# - debug : limit batches, fewer epochs (quick sanity check)
# - smoke : 1 epoch, very few batches (pipeline check)
TRAINING_MODE = "full"

MAX_TRAIN_BATCHES = None  # set automatically from TRAINING_MODE if needed
MAX_VAL_BATCHES = None

# Convenience presets (applied in Trainer if TRAINING_MODE != "full")
TRAIN_MODE_PRESETS = {
    "smoke": {
        "max_train_batches": 2,
        "max_val_batches": 1,
        "epochs": 1,
    },
    "debug": {
        "max_train_batches": 20,
        "max_val_batches": 5,
        "epochs": 2,
    },
    "full": {
        "max_train_batches": None,
        "max_val_batches": None,
        "epochs": NUM_EPOCHS,
    },
}

# ============================================================
# Early Stopping
# ============================================================

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.0001
MONITOR_METRIC = "val_loss"

# ============================================================
# Checkpoints
# ============================================================

SAVE_LAST = True
SAVE_BEST = True
SAVE_EVERY_N_EPOCHS = 5

BEST_MODEL_NAME = "best_model.pt"
LAST_MODEL_NAME = "last_model.pt"

RESUME_TRAINING = False
RESUME_CHECKPOINT = None

# ============================================================
# History & Plots
# ============================================================

SAVE_HISTORY = True
SAVE_PLOTS = True

# ============================================================
# Logging
# ============================================================

LOG_INTERVAL = 10
VERBOSE = True

# ============================================================
# Evaluation
# ============================================================

CONFUSION_MATRIX = True
CLASSIFICATION_REPORT = True
SAVE_PREDICTIONS = True

# ============================================================
# Metadata Files
# ============================================================

CLASS_NAMES_FILE = METADATA_DIR / "class_names.json"
DATASET_INFO_FILE = METADATA_DIR / "dataset_info.json"
TRAINING_HISTORY_FILE = METADATA_DIR / "training_history.json"

# ============================================================
# Environment Variables
# ============================================================

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# ============================================================
# API (used later by FastAPI)
# ============================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

TOP_K_PREDICTIONS = 3
CONFIDENCE_THRESHOLD = 0.50
