"""
B1: Train SmallCNN from scratch on CIFAR-10 subset.
Generates figures/cnn_curves.png
Saves model and history with smart fallback.
"""
import os
import time
import torch
from torch import nn
from loguru import logger

from src.data import get_cifar10_subset, make_loaders, set_seed
from src.engine import evaluate, fit
from src.models import SmallCNN
from src.utils import history_exists, load_history, plot_history, save_history


def train_smallcnn():
    """Train SmallCNN and return history."""
    logger.info("=" * 50)
    logger.info("B1: Training SmallCNN")
    logger.info("=" * 50)

    STUDENT_ID = 121314
    set_seed(STUDENT_ID)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    classes = [0, 1, 2, 3, 4]
    batch_size = 64
    epochs = 20
    learning_rate = 1e-3

    logger.debug(f"Classes: {classes}, batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    # Load data
    logger.info("Loading CIFAR-10 subset...")
    train_ds, val_ds, test_ds = get_cifar10_subset(
        root="./data",
        classes=classes,
        n_train_per_class=800,
        n_test_per_class=200,
        image_size=32,
        imagenet_norm=False,
        seed=STUDENT_ID
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create model
    logger.info("Creating SmallCNN model...")
    model = SmallCNN(num_classes=len(classes)).to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    history = fit(
        model, train_loader, val_loader, epochs,
        optimizer, loss_fn, device, mix=None
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/smallcnn.pth")
    logger.success("Model saved to models/smallcnn.pth")

    # Save history
    save_history(history, "models/smallcnn_history.csv")

    # Generate figure
    plot_history(history, "SmallCNN Training", "figures/cnn_curves.png")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    logger.success(f"Test Accuracy: {test_metrics['acc']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"B1 completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return history


def get_smallcnn_history():
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    """
    history_path = "models/smallcnn_history.csv"

    if history_exists(history_path):
        logger.info("📂 SmallCNN history found. Loading...")
        return load_history(history_path)
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