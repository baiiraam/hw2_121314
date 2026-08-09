"""
B2: Compare ResNet-18 feature extraction vs fine-tuning.
Generates figures/transfer_compare.png
Saves best model for B3 (model persistence).
Uses smart fallback for SmallCNN history.
"""
import os

import torch
from torch import nn

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
    print(f"\n=== Training ResNet-18 ({mode}) ===")

    model = build_resnet18(num_classes=num_classes, mode=mode)
    model = model.to(device)

    trainable = count_trainable_params(model)
    print(f"Trainable params: {trainable}")

    # Smaller LR for fine-tuning (Problem 6b)
    lr = 1e-4 if mode == "finetune" else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = fit(
        model, train_loader, val_loader, 20,
        optimizer, loss_fn, device, mix=None
    )

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    test_acc = test_metrics["acc"]
    print(f"Test Accuracy: {test_acc:.4f}")

    return history, test_acc, model


def get_resnet_history(mode, train_func):
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    """
    history_path = f"models/{mode}_history.csv"
    model_path = f"models/{mode}_model.pth"

    if history_exists(history_path) and os.path.exists(model_path):
        print(f"📂 {mode} history found. Loading...")
        return load_history(history_path), None
    else:
        print(f"🔨 {mode} history not found. Training from scratch...")
        history, _test_acc, model = train_func()
        save_history(history, history_path)
        torch.save(model.state_dict(), model_path)
        return history, model


def get_smallcnn_history():
    """
    Smart fallback for SmallCNN history.
    If not found, import and run train_cnn's training function.
    """
    history_path = "models/smallcnn_history.csv"

    if history_exists(history_path):
        print("📂 SmallCNN history found. Loading...")
        return load_history(history_path)
    else:
        print("🔨 SmallCNN history not found. Training from scratch...")
        from experiments.train_cnn import train_smallcnn
        return train_smallcnn()


def main():
    STUDENT_ID = 121314
    set_seed(STUDENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = [0, 1, 2, 3, 4]
    batch_size = 64
    num_classes = len(classes)

    # Load data (ResNet: 224x224, ImageNet normalization)
    train_ds, val_ds, test_ds = get_cifar10_subset(
        root="./data",
        classes=classes,
        n_train_per_class=800,
        n_test_per_class=200,
        image_size=224,
        imagenet_norm=True,
        seed=STUDENT_ID
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    os.makedirs("models", exist_ok=True)

    # === Feature Extraction ===
    def train_fe():
        return train_resnet("feature_extract", train_loader, val_loader,
                           test_loader, num_classes, device)

    history_fe, _model_fe = get_resnet_history("feature_extract", train_fe)

    # === Fine-Tuning ===
    def train_ft():
        return train_resnet("finetune", train_loader, val_loader,
                           test_loader, num_classes, device)

    history_ft, model_ft = get_resnet_history("finetune", train_ft)

    # === Get SmallCNN history (with smart fallback) ===
    history_cnn = get_smallcnn_history()

    # === Generate comparison figure ===
    plot_transfer_comparison(
        history_cnn, history_fe, history_ft,
        "figures/transfer_compare.png"
    )

    # === Save best model for B3 ===
    if model_ft is not None:
        torch.save(model_ft.state_dict(), "models/best_resnet_ft.pth")
        print("✅ Best model saved to models/best_resnet_ft.pth")
    else:
        # If model_ft was loaded from checkpoint, we need to recreate it
        # (In practice, this is handled by augment_compare loading from file)
        print("ℹ️  Model loaded from checkpoint - best model file should already exist")


if __name__ == "__main__":
    main()