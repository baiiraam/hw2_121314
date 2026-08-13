"""
B1: Train SmallCNN from scratch on CIFAR-10 subset.
Generates figures/cnn_curves.png
Saves model and history with smart fallback.
"""

import os
import time

import torch
from loguru import logger
from torch import nn

from src.config import (
    BATCH_SIZE,
    CLASSES,
    CNN_CURVES,
    DATA_ROOT,
    EPOCHS,
    LR_SMALLCNN,
    MODELS_DIR,
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    SMALLCNN_CHECKPOINT,
    SMALLCNN_HISTORY,
    STUDENT_ID,
)
from src.data import get_cifar10_subset, make_loaders, set_seed
from src.engine import evaluate, fit
from src.models import SmallCNN
from src.utils import history_exists, load_history, plot_history, save_history


def train_smallcnn():
    """Train SmallCNN and return history."""
    logger.info("=" * 50)
    logger.info("B1: Training SmallCNN")
    logger.info("=" * 50)

    set_seed(STUDENT_ID)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.debug(
        f"Classes: {CLASSES}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR_SMALLCNN}"
    )

    # Load data
    logger.info("Loading CIFAR-10 subset...")
    train_ds, val_ds, test_ds = get_cifar10_subset(
        root=DATA_ROOT,
        classes=CLASSES,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        image_size=32,
        imagenet_norm=False,
        seed=STUDENT_ID,
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=BATCH_SIZE
    )
    logger.info(
        f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches"
    )

    # Create model
    logger.info("Creating SmallCNN model...")
    model = SmallCNN(num_classes=len(CLASSES)).to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    history = fit(
        model, train_loader, val_loader, EPOCHS, optimizer, loss_fn, device, mix=None
    )

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), SMALLCNN_CHECKPOINT)
    logger.success(f"Model saved to {SMALLCNN_CHECKPOINT}")

    # Save history
    save_history(history, SMALLCNN_HISTORY)

    # Generate figure
    plot_history(history, "SmallCNN Training", CNN_CURVES)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    logger.success(f"Test Accuracy: {test_metrics['acc']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"B1 completed in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    return history


def get_smallcnn_history():
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    """
    if history_exists(SMALLCNN_HISTORY):
        logger.info("📂 SmallCNN history found. Loading...")
        return load_history(SMALLCNN_HISTORY)
    else:
        logger.warning("🔨 SmallCNN history not found. Training from scratch...")
        return train_smallcnn()


def main():
    """Main entry point with smart fallback."""
    logger.info("Starting B1: SmallCNN")
    history = get_smallcnn_history()
    logger.success(f"SmallCNN ready: {len(history['train_loss'])} epochs available")


if __name__ == "__main__":
    main()
