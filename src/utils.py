"""
Utility functions: accuracy, parameter counting, plotting, history I/O.
"""
import csv
import os

import matplotlib.pyplot as plt


def accuracy(logits, targets):
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def count_trainable_params(model):
    """
    Count parameters with requires_grad=True.

    Used for Part C2: feature_extract (~2,565) vs finetune (~11.2M)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_history(history, filepath):
    """
    Save training history to CSV file.

    Args:
        history: dict with keys: train_loss, train_acc, val_loss, val_acc
        filepath: path to save CSV file
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        n_epochs = len(history["train_loss"])
        for epoch in range(n_epochs):
            writer.writerow([
                epoch,
                history["train_loss"][epoch],
                history["train_acc"][epoch],
                history["val_loss"][epoch],
                history["val_acc"][epoch],
            ])

    print(f"✅ History saved to {filepath}")


def load_history(filepath):
    """
    Load training history from CSV file.

    Returns:
        dict with keys: train_loss, train_acc, val_loss, val_acc
    """
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

    print(f"✅ History loaded from {filepath} ({len(history['train_loss'])} epochs)")
    return history


def history_exists(filepath):
    """Check if history file exists and is not empty."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def plot_history(history, title, save_path):
    """Generate training curves (loss and accuracy) and save to file."""
    epochs = range(1, len(history["train_loss"]) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Validation", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
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
    print(f"✅ Figure saved to {save_path}")


def plot_transfer_comparison(cnn_history, fe_history, ft_history, save_path):
    """Plot validation accuracy for SmallCNN, feature extraction, and fine-tuning."""
    epochs = range(1, len(cnn_history["val_acc"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cnn_history["val_acc"],
             label="SmallCNN (from scratch)", marker="o", markersize=4)
    plt.plot(epochs, fe_history["val_acc"],
             label="ResNet-18 (feature extraction)", marker="s", markersize=4)
    plt.plot(epochs, ft_history["val_acc"],
             label="ResNet-18 (fine-tuning)", marker="^", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Transfer Learning Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to {save_path}")


def plot_augment_comparison(none_history, std_history, mix_history, save_path):
    """Plot validation accuracy for three augmentation regimes."""
    epochs = range(1, len(none_history["val_acc"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, none_history["val_acc"],
             label="No Augmentation", marker="o", markersize=4)
    plt.plot(epochs, std_history["val_acc"],
             label="Standard (Crop + Flip)", marker="s", markersize=4)
    plt.plot(epochs, mix_history["val_acc"],
             label="Mixup", marker="^", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Augmentation Study Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to {save_path}")