"""
Utility functions: accuracy, parameter counting, plotting, history I/O.
"""

import csv
import os
import time

import matplotlib.pyplot as plt
from loguru import logger


def accuracy(logits, targets):
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def count_trainable_params(model):
    """
    Count parameters with requires_grad=True.

    Used for Part C2: feature_extract (~2,565) vs finetune (~11.2M)
    """
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"Trainable parameters: {count:,}")
    return count


def get_file_size(filepath: str) -> str:
    """Get human-readable file size."""
    if not os.path.exists(filepath):
        return "N/A"
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def save_history(history, filepath):
    """
    Save training history to CSV file.
    """
    logger.debug(f"Saving history to {filepath}")
    start = time.time()

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        n_epochs = len(history["train_loss"])
        for epoch in range(n_epochs):
            writer.writerow(
                [
                    epoch,
                    history["train_loss"][epoch],
                    history["train_acc"][epoch],
                    history["val_loss"][epoch],
                    history["val_acc"][epoch],
                ]
            )

    elapsed = time.time() - start
    logger.success(f"History saved to {filepath} ({n_epochs} epochs, {elapsed:.2f}s)")


def load_history(filepath):
    """
    Load training history from CSV file.

    Returns:
        dict with keys: train_loss, train_acc, val_loss, val_acc
    """
    logger.debug(f"Loading history from {filepath}")
    start = time.time()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["train_loss"].append(float(row["train_loss"]))
            history["train_acc"].append(float(row["train_acc"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["val_acc"].append(float(row["val_acc"]))

    elapsed = time.time() - start
    n_epochs = len(history["train_loss"])
    logger.success(
        f"History loaded from {filepath} ({n_epochs} epochs, {elapsed:.2f}s)"
    )
    return history


def history_exists(filepath):
    """Check if history file exists and is not empty."""
    exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    if exists:
        logger.debug(f"History exists: {filepath} ({get_file_size(filepath)})")
    else:
        logger.debug(f"History not found: {filepath}")
    return exists


def plot_history(history, title, save_path):
    """Generate training curves (loss and accuracy) and save to file."""
    logger.debug(f"Plotting history to {save_path}")
    start = time.time()

    epochs = range(1, len(history["train_loss"]) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Validation", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Validation", marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs. Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start
    logger.success(f"Figure saved to {save_path} in {elapsed:.2f}s")


def plot_transfer_comparison(cnn_history, fe_history, ft_history, save_path):
    """Plot validation accuracy for SmallCNN, feature extraction, and fine-tuning."""
    logger.debug(f"Plotting transfer comparison to {save_path}")
    start = time.time()

    epochs = range(1, len(cnn_history["val_acc"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        cnn_history["val_acc"],
        label="SmallCNN (from scratch)",
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        fe_history["val_acc"],
        label="ResNet-18 (feature extraction)",
        marker="s",
        markersize=4,
    )
    plt.plot(
        epochs,
        ft_history["val_acc"],
        label="ResNet-18 (fine-tuning)",
        marker="^",
        markersize=4,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Transfer Learning Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start
    logger.success(f"Transfer comparison figure saved to {save_path} in {elapsed:.2f}s")


def plot_augment_comparison(none_history, std_history, mix_history, save_path):
    """Plot validation accuracy for three augmentation regimes."""
    logger.debug(f"Plotting augmentation comparison to {save_path}")
    start = time.time()

    epochs = range(1, len(none_history["val_acc"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        none_history["val_acc"],
        label="No Augmentation",
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        std_history["val_acc"],
        label="Standard (Crop + Flip)",
        marker="s",
        markersize=4,
    )
    plt.plot(epochs, mix_history["val_acc"], label="Mixup", marker="^", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Augmentation Study Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start
    logger.success(
        f"Augmentation comparison figure saved to {save_path} in {elapsed:.2f}s"
    )
