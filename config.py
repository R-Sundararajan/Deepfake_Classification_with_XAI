"""Single source of truth for the root-level review prototype."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 2
NUM_WORKERS = 4

MODEL_NAME = "resnet18"
NUM_CLASSES = 2
CLASS_NAMES = ("fake", "real")
TRAINABLE_BACKBONE_BLOCKS = 1

DATASET_PATH = PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake"
TRAIN_PATH = DATASET_PATH / "train"
VALID_PATH = DATASET_PATH / "valid"
TEST_PATH = DATASET_PATH / "test"

MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "best_resnet18.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
