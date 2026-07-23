"""
Project Configuration File

This file contains all global settings used throughout the project.
Changing a value here updates it everywhere in the project.
"""

import torch
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"

# Dataset folder
DATASET_DIR = DATA_DIR / "real_vs_fake" / "real-vs-fake"

# Dataset splits
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

# Output folders
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# DATASET SETTINGS
# ============================================================

IMAGE_SIZE = 256
NUM_CHANNELS = 3
NUM_CLASSES = 2

# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# MODEL SETTINGS
# ============================================================

PATCH_SIZE = 16

# Recommended for RTX 3050 Laptop GPU
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
MLP_DIM = 512

DROPOUT = 0.1

# ============================================================
# SYSTEM SETTINGS
# ============================================================

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# DATALOADER SETTINGS
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = True

SHUFFLE_TRAIN = True
SHUFFLE_VALID = False
SHUFFLE_TEST = False