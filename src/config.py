# =====================================================================
# FILE: src/config.py
# =====================================================================
"""
Centralized configuration for all HW2 experiments.

All hyperparameters, paths, and settings are defined here.
Import from this file instead of hardcoding values.
"""

import os

# ============================================================
# Student Information
# ============================================================
STUDENT_ID = 121314
STUDENT_NAME = "Bayram Bayramov"
STUDENT_GROUP = "DL"
STUDENT_EMAIL = "bayramovb578@gmail.com"

# ============================================================
# Data Configuration
# ============================================================
CLASSES = [0, 1, 2, 3, 4]  # Airplane, Automobile, Bird, Cat, Deer
BATCH_SIZE = 64
N_TRAIN_PER_CLASS = 800
N_TEST_PER_CLASS = 200
VAL_SIZE_PER_CLASS = 160  # 20% of 800 (640 train, 160 val per class)
DATA_ROOT = "./data"

# ============================================================
# Training Configuration
# ============================================================
EPOCHS = 20

# Learning rates
LR_SMALLCNN = 1e-3
LR_FEATURE_EXTRACT = 1e-3
LR_FINETUNE = 1e-4  # Smaller LR for fine-tuning (Problem 6b)

# Optimizer
OPTIMIZER = "adam"  # "adam" or "sgd"

# ============================================================
# Augmentation Configuration
# ============================================================
MIXUP_ALPHA = 1.0  # Beta distribution parameter for Mixup
CUTMIX_ALPHA = 1.0  # Beta distribution parameter for CutMix

# Standard augmentation parameters
RANDOM_CROP_PADDING = 8
RANDOM_HORIZONTAL_FLIP_PROB = 0.5

# ============================================================
# Model Configuration
# ============================================================
NUM_CLASSES = len(CLASSES)
SMALLCNN_CHANNELS = [32, 64, 128]  # Channels per block

# ResNet
RESNET_IMAGE_SIZE = 224
RESNET_NORMALIZATION = "imagenet"  # "imagenet" or "cifar10"

# ============================================================
# Paths
# ============================================================
MODELS_DIR = "./models"
FIGURES_DIR = "./figures"
LOGS_DIR = "./logs"
CACHE_DIR = os.path.join(DATA_ROOT, "cache")
SAMPLES_DIR = os.path.join(DATA_ROOT, "samples")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

# ============================================================
# Checkpoint Files
# ============================================================
SMALLCNN_CHECKPOINT = os.path.join(MODELS_DIR, "smallcnn.pth")
SMALLCNN_HISTORY = os.path.join(MODELS_DIR, "smallcnn_history.csv")

FEATURE_EXTRACT_CHECKPOINT = os.path.join(MODELS_DIR, "feature_extract_model.pth")
FEATURE_EXTRACT_HISTORY = os.path.join(MODELS_DIR, "feature_extract_history.csv")

FINETUNE_CHECKPOINT = os.path.join(MODELS_DIR, "finetune_model.pth")
FINETUNE_HISTORY = os.path.join(MODELS_DIR, "finetune_history.csv")

BEST_RESNET_CHECKPOINT = os.path.join(MODELS_DIR, "best_resnet_ft.pth")

AUGMENT_NONE_CHECKPOINT = os.path.join(MODELS_DIR, "augment_none.pth")
AUGMENT_NONE_HISTORY = os.path.join(MODELS_DIR, "none_history.csv")

AUGMENT_STD_CHECKPOINT = os.path.join(MODELS_DIR, "augment_std.pth")
AUGMENT_STD_HISTORY = os.path.join(MODELS_DIR, "std_history.csv")

AUGMENT_MIX_CHECKPOINT = os.path.join(MODELS_DIR, "augment_mix.pth")
AUGMENT_MIX_HISTORY = os.path.join(MODELS_DIR, "mix_history.csv")

# ============================================================
# Figure Files
# ============================================================
CNN_CURVES = os.path.join(FIGURES_DIR, "cnn_curves.png")
TRANSFER_COMPARE = os.path.join(FIGURES_DIR, "transfer_compare.png")
AUGMENT_COMPARE = os.path.join(FIGURES_DIR, "augment_compare.png")
ZOO_PREFIX = os.path.join(FIGURES_DIR, "zoo_")

# ============================================================
# Bonus Inference
# ============================================================
BONUS_SCORE_THRESHOLD = 0.5
