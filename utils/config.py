
"""
Project Configuration File
"""

import torch
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "real_vs_fake" / "real-vs-fake"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

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

BATCH_SIZE = 64

# Change this to 6 if you want exactly 6 total epochs.
EPOCHS = 5
MAX_EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# MODEL SETTINGS
# ============================================================

PATCH_SIZE = 16

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
# CUDA OPTIMIZATION
# ============================================================

if DEVICE.type == "cuda":

    # Allows TF32 operations on supported NVIDIA GPUs.
    # RTX 4060 supports this.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Useful when image dimensions stay fixed at 256x256.
    torch.backends.cudnn.benchmark = True

# ============================================================
# DATALOADER SETTINGS
# ============================================================

# IMPORTANT:
# 0 = data loading happens in the training process.
# 4 gives the GPU workers that can prepare batches ahead of time.
#
# If your CPU has plenty of cores, try 6 or 8.
NUM_WORKERS = 4

PIN_MEMORY = True

SHUFFLE_TRAIN = True
SHUFFLE_VALID = False
SHUFFLE_TEST = False
