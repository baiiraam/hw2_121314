"""
B2: Compare ResNet-18 feature extraction vs fine-tuning.
Generates figures/transfer_compare.png
Saves best model for B3 (model persistence).
Uses smart fallback for SmallCNN history.
"""

import os
import time

import torch
from loguru import logger
from torch import nn

from src.config import (
    BATCH_SIZE,
    BEST_RESNET_CHECKPOINT,
    CLASSES,
    DATA_ROOT,
    EPOCHS,
    FEATURE_EXTRACT_CHECKPOINT,
    FEATURE_EXTRACT_HISTORY,
    FINETUNE_CHECKPOINT,
    FINETUNE_HISTORY,
    LR_FEATURE_EXTRACT,
    LR_FINETUNE,
    MODELS_DIR,
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    SMALLCNN_HISTORY,
    STUDENT_ID,
    TRANSFER_COMPARE,
)
from src.data import get_cifar10_subset, make_loaders, set_seed
from src.engine import evaluate, fit
from src.models import build_resnet18
from src.utils import (
    count_trainable_params,
    history_exists,
    load_history,
    plot_transfer_comparison,
    save_history,
)


def train_resnet(mode, train_loader, val_loader, test_loader, num_classes, device):
    """Train ResNet-18 in specified mode and return history and test accuracy."""
    logger.info(f"\n=== Training ResNet-18 ({mode}) ===")

    model = build_resnet18(num_classes=num_classes, mode=mode)
    model = model.to(device)

    trainable = count_trainable_params(model)
    logger.info(f"Trainable params: {trainable:,}")

    # Smaller LR for fine-tuning (Problem 6b)
    lr = LR_FINETUNE if mode == "finetune" else LR_FEATURE_EXTRACT
    logger.debug(f"Using learning rate: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = fit(
        model, train_loader, val_loader, EPOCHS, optimizer, loss_fn, device, mix=None
    )

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    test_acc = test_metrics["acc"]
    logger.success(f"Test Accuracy: {test_acc:.4f}")

    return history, test_acc, model


def get_resnet_history(mode, train_func):
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    """
    history_path = (
        FEATURE_EXTRACT_HISTORY if mode == "feature_extract" else FINETUNE_HISTORY
    )
    model_path = (
        FEATURE_EXTRACT_CHECKPOINT if mode == "feature_extract" else FINETUNE_CHECKPOINT
    )

    if history_exists(history_path) and os.path.exists(model_path):
        logger.info(f"📂 {mode} history found. Loading...")
        return load_history(history_path), None
    else:
        logger.warning(f"🔨 {mode} history not found. Training from scratch...")
        history, _test_acc, model = train_func()
        save_history(history, history_path)
        torch.save(model.state_dict(), model_path)
        logger.success(f"Saved {mode} model to {model_path}")
        return history, model


def get_smallcnn_history():
    """
    Smart fallback for SmallCNN history.
    If not found, import and run train_cnn's training function.
    """
    if history_exists(SMALLCNN_HISTORY):
        logger.info("📂 SmallCNN history found. Loading...")
        return load_history(SMALLCNN_HISTORY)
    else:
        logger.warning("🔨 SmallCNN history not found. Training from scratch...")
        from experiments.train_cnn import train_smallcnn

        return train_smallcnn()


def main():
    logger.info("=" * 50)
    logger.info("B2: Transfer Learning Comparison")
    logger.info("=" * 50)

    start_time = time.time()

    set_seed(STUDENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.debug(f"Classes: {CLASSES}, batch_size={BATCH_SIZE}")

    # Load data (ResNet: 224x224, ImageNet normalization)
    logger.info("Loading CIFAR-10 subset (224x224, ImageNet norm)...")
    train_ds, val_ds, test_ds = get_cifar10_subset(
        root=DATA_ROOT,
        classes=CLASSES,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        image_size=224,
        imagenet_norm=True,
        seed=STUDENT_ID,
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=BATCH_SIZE
    )

    os.makedirs(MODELS_DIR, exist_ok=True)

    # === Feature Extraction ===
    def train_fe():
        return train_resnet(
            "feature_extract",
            train_loader,
            val_loader,
            test_loader,
            len(CLASSES),
            device,
        )

    history_fe, _model_fe = get_resnet_history("feature_extract", train_fe)

    # === Fine-Tuning ===
    def train_ft():
        return train_resnet(
            "finetune", train_loader, val_loader, test_loader, len(CLASSES), device
        )

    history_ft, model_ft = get_resnet_history("finetune", train_ft)

    # === Get SmallCNN history (with smart fallback) ===
    history_cnn = get_smallcnn_history()

    # === Generate comparison figure ===
    plot_transfer_comparison(history_cnn, history_fe, history_ft, TRANSFER_COMPARE)

    # === Save best model for B3 ===
    if model_ft is not None:
        torch.save(model_ft.state_dict(), BEST_RESNET_CHECKPOINT)
        logger.success(f"Best model saved to {BEST_RESNET_CHECKPOINT}")
    else:
        logger.info(
            "ℹ️  Model loaded from checkpoint - best model file should already exist"
        )

    elapsed = time.time() - start_time
    logger.info(f"B2 completed in {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
