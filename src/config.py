"""
Configuration module for the Manufacturing Defect Detection System.

Centralizes all hyperparameters, paths, and settings for reproducibility.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["defective", "normal"]   # alphabetical: 0 = defective, 1 = normal
NUM_CLASSES = 2
IMG_SIZE = 224                          # input resolution for the model
RANDOM_SEED = 42

# Split ratios (train / val / test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
BACKBONE = "mobilenetv3_small_100"      # timm model identifier
PRETRAINED = True
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"                # "cosine" | "step"
STEP_SIZE = 7                           # for StepLR
GAMMA = 0.1                            # for StepLR

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

# Number of data-loader workers
NUM_WORKERS = int(os.cpu_count() or 4) // 2

# ──────────────────────────────────────────────────────────────────────────────
# Model Artifact Names
# ──────────────────────────────────────────────────────────────────────────────
BEST_MODEL_FILENAME = "best_model.pth"
ONNX_MODEL_FILENAME = "model.onnx"

# ──────────────────────────────────────────────────────────────────────────────
# API Settings
# ──────────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = MODEL_DIR / BEST_MODEL_FILENAME
CONFIDENCE_THRESHOLD = 0.5
