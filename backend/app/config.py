import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 configuration
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Attack configuration defaults
DEFAULT_FGSM_EPS = 0.03
DEFAULT_PGD_EPS = 0.03
DEFAULT_PGD_ALPHA = 0.007
DEFAULT_PGD_STEPS = 10

# Backdoor configuration defaults
DEFAULT_TRIGGER_TYPE = "checkerboard_3x3"
DEFAULT_TARGET_CLASS = 0  # 'airplane'
TRIGGER_POSITION = (28, 28)  # Bottom-right for 32x32 image
TRIGGER_SIZE = 3

# Mahalanobis Detector defaults
MAHALANOBIS_CALIBRATION_FPR = 0.05  # 95% clean TPR calibration threshold
MAHALANOBIS_LAYER = "layer4"

# Checkpoint paths
CLEAN_MODEL_PATH = CHECKPOINT_DIR / "resnet18_cifar10_clean.pt"
BACKDOOR_MODEL_PATH = CHECKPOINT_DIR / "resnet18_cifar10_backdoor.pt"
MAHALANOBIS_STATS_PATH = CHECKPOINT_DIR / "mahalanobis_stats.pt"
