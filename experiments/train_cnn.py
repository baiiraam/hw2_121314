"""
B1: Train SmallCNN from scratch on CIFAR-10 subset.
Generates figures/cnn_curves.png
Saves model and history with smart fallback.
"""
import os

import torch
from torch import nn

from src.data import get_cifar10_subset, make_loaders, set_seed
from src.engine import evaluate, fit
from src.models import SmallCNN
from src.utils import history_exists, load_history, plot_history, save_history


def train_smallcnn():
    """Train SmallCNN and return history."""
    STUDENT_ID = 121314
    set_seed(STUDENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = [0, 1, 2, 3, 4]
    batch_size = 64
    epochs = 20
    learning_rate = 1e-3

    # Load data (SmallCNN: 32x32, CIFAR-10 normalization)
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

    # Create model
    model = SmallCNN(num_classes=len(classes)).to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print("\n=== B1: Training SmallCNN ===")
    history = fit(
        model, train_loader, val_loader, epochs,
        optimizer, loss_fn, device, mix=None
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/smallcnn.pth")
    print("✅ Model saved to models/smallcnn.pth")

    # Save history
    save_history(history, "models/smallcnn_history.csv")

    # Generate figure
    plot_history(history, "SmallCNN Training", "figures/cnn_curves.png")

    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")

    return history


def get_smallcnn_history():
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    """
    history_path = "models/smallcnn_history.csv"

    if history_exists(history_path):
        print("📂 SmallCNN history found. Loading...")
        return load_history(history_path)
    else:
        print("🔨 SmallCNN history not found. Training from scratch...")
        return train_smallcnn()


def main():
    """Main entry point with smart fallback."""
    history = get_smallcnn_history()
    print(f"\n✅ SmallCNN ready: {len(history['train_loss'])} epochs available")


if __name__ == "__main__":
    main()