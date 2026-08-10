"""
B3: Data augmentation study.
Compares: (a) no augmentation, (b) standard (crop + flip), (c) Mixup.
Generates figures/augment_compare.png
Loads best model from B2 (model persistence).
Saves all histories with smart fallback.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset
from torchvision import datasets
from loguru import logger

from src.data import set_seed, get_cifar10_subset, make_loaders
from src.models import build_resnet18
from src.engine import fit, evaluate
from src.utils import (
    plot_augment_comparison,
    save_history,
    load_history,
    history_exists,
)


def train_regime(model, train_loader, val_loader, device, regime_name="none", mix=None):
    """
    Train a regime and return history and the trained model.

    Args:
        model: Model to train (starts from B2 checkpoint)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch.device
        regime_name: Name of the regime for logging ("none", "std", "mixup")
        mix: "mixup" or None (passed to engine.fit)
    """
    mix_str = mix if mix else "none"
    logger.debug(f"Training regime: {regime_name} (mix={mix_str})")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = fit(
        model, train_loader, val_loader, 20,
        optimizer, loss_fn, device, mix=mix
    )

    return history, model


def get_regime_history(regime_name, train_func):
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    Also saves the trained model for later evaluation.
    """
    history_path = f"models/{regime_name}_history.csv"
    model_path = f"models/augment_{regime_name}.pth"

    if history_exists(history_path) and os.path.exists(model_path):
        logger.info(f"📂 {regime_name} history found. Loading...")
        history = load_history(history_path)
        return history, model_path
    else:
        logger.warning(f"🔨 {regime_name} history not found. Training from scratch...")
        history, model = train_func()
        save_history(history, history_path)
        torch.save(model.state_dict(), model_path)
        logger.success(f"Saved {regime_name} model to {model_path}")
        return history, model_path


def load_base_model(device):
    """Load the best model from B2."""
    logger.debug("Loading best model from B2...")
    model = build_resnet18(num_classes=5, mode="finetune")
    model.load_state_dict(torch.load("models/best_resnet_ft.pth", map_location=device))
    model = model.to(device)
    logger.debug("Best model loaded successfully")
    return model


def load_regime_model(regime_name, device):
    """Load a regime-specific trained model."""
    model_path = f"models/augment_{regime_name}.pth"
    logger.debug(f"Loading {regime_name} model from {model_path}...")
    model = build_resnet18(num_classes=5, mode="finetune")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    logger.debug(f"{regime_name} model loaded successfully")
    return model


def main():
    logger.info("=" * 50)
    logger.info("B3: Augmentation Study")
    logger.info("=" * 50)

    start_time = time.time()

    STUDENT_ID = 121314
    set_seed(STUDENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    classes = [0, 1, 2, 3, 4]
    batch_size = 64
    num_classes = len(classes)

    logger.debug(f"Classes: {classes}, batch_size={batch_size}")

    # Load base data (ResNet: 224x224, ImageNet normalization)
    logger.info("Loading CIFAR-10 subset...")
    base_train_ds, val_ds, test_ds = get_cifar10_subset(
        root="./data",
        classes=classes,
        n_train_per_class=800,
        n_test_per_class=200,
        image_size=224,
        imagenet_norm=True,
        seed=STUDENT_ID
    )

    os.makedirs("models", exist_ok=True)

    # ================================================================
    # Regime 1: No augmentation
    # ================================================================
    logger.info("\n=== B3: No Augmentation ===")
    train_loader_no_aug, val_loader_no_aug, test_loader = make_loaders(
        base_train_ds, val_ds, test_ds, batch_size=batch_size
    )

    def train_none():
        model = load_base_model(device)
        return train_regime(
            model,
            train_loader_no_aug,
            val_loader_no_aug,
            device,
            regime_name="none",
            mix=None
        )

    history_none, model_none_path = get_regime_history("none", train_none)

    # ================================================================
    # Regime 2: Standard augmentation (crop + flip)
    # ================================================================
    logger.info("\n=== B3: Standard Augmentation (Crop + Flip) ===")
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Load CIFAR-10 with augmentation transforms
    full_train = datasets.CIFAR10(
        root="./data", train=True, download=False,
        transform=augment_transform
    )

    def filter_by_class(dataset, class_list):
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_list]
        return Subset(dataset, indices)

    train_filtered = filter_by_class(full_train, classes)

    # Use same seed for reproducible split
    rng = np.random.RandomState(STUDENT_ID)

    # Group indices by class
    train_indices_by_class = {c: [] for c in classes}
    for idx, (_, label) in enumerate(train_filtered):
        if label in classes:
            train_indices_by_class[label].append(idx)

    # Select 800 samples per class (same as base dataset)
    train_selected = []
    for c in classes:
        indices = train_indices_by_class[c]
        rng.shuffle(indices)
        train_selected.extend(indices[:800])

    # Split into train (640) and val (160) per class
    val_size_per_class = 160
    train_size_per_class = 800 - val_size_per_class
    train_indices = []
    val_indices = []

    for c in classes:
        class_indices = [idx for idx in train_selected
                         if train_filtered[idx][1] == c]
        rng.shuffle(class_indices)
        train_indices.extend(class_indices[:train_size_per_class])
        val_indices.extend(class_indices[train_size_per_class:800])

    train_ds_aug = Subset(train_filtered, train_indices)
    val_ds_aug = Subset(train_filtered, val_indices)

    train_loader_std, val_loader_std, _ = make_loaders(
        train_ds_aug, val_ds_aug, test_ds, batch_size=batch_size
    )

    def train_std():
        model = load_base_model(device)
        return train_regime(
            model,
            train_loader_std,
            val_loader_std,
            device,
            regime_name="std",
            mix=None
        )

    history_std, model_std_path = get_regime_history("std", train_std)

    # ================================================================
    # Regime 3: Mixup
    # ================================================================
    logger.info("\n=== B3: Mixup Augmentation ===")

    def train_mix():
        model = load_base_model(device)
        return train_regime(
            model,
            train_loader_std,      # Reuse the standard augmentation DataLoader
            val_loader_std,
            device,
            regime_name="mixup",
            mix="mixup"            # ← This enables Mixup in the training loop
        )

    history_mix, model_mix_path = get_regime_history("mix", train_mix)

    # ================================================================
    # Generate comparison figure
    # ================================================================
    plot_augment_comparison(
        history_none, history_std, history_mix,
        "figures/augment_compare.png"
    )

    # ================================================================
    # Evaluate all regimes on test set
    # ================================================================
    logger.info("\n=== B3: Test Results ===")

    # Load the ACTUAL trained models for each regime (not the B2 base)
    model_none = load_regime_model("none", device)
    model_std = load_regime_model("std", device)
    model_mix = load_regime_model("mix", device)

    loss_fn = nn.CrossEntropyLoss()
    test_metrics_none = evaluate(model_none, test_loader, loss_fn, device)
    test_metrics_std = evaluate(model_std, test_loader, loss_fn, device)
    test_metrics_mix = evaluate(model_mix, test_loader, loss_fn, device)

    logger.success(f"No Augmentation:  Test Acc = {test_metrics_none['acc']:.4f}")
    logger.success(f"Standard:         Test Acc = {test_metrics_std['acc']:.4f}")
    logger.success(f"Mixup:            Test Acc = {test_metrics_mix['acc']:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"B3 completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()