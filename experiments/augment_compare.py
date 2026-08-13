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
from loguru import logger
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.config import (
    AUGMENT_COMPARE,
    AUGMENT_MIX_CHECKPOINT,
    AUGMENT_MIX_HISTORY,
    AUGMENT_NONE_CHECKPOINT,
    AUGMENT_NONE_HISTORY,
    AUGMENT_STD_CHECKPOINT,
    AUGMENT_STD_HISTORY,
    BATCH_SIZE,
    BEST_RESNET_CHECKPOINT,
    CLASSES,
    DATA_ROOT,
    EPOCHS,
    LR_FINETUNE,
    MODELS_DIR,
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
    RANDOM_CROP_PADDING,
    RANDOM_HORIZONTAL_FLIP_PROB,
    STUDENT_ID,
)
from src.data import get_cifar10_subset, make_loaders, set_seed
from src.engine import evaluate, fit
from src.models import build_resnet18
from src.utils import (
    history_exists,
    load_history,
    plot_augment_comparison,
    save_history,
)


def train_regime(model, train_loader, val_loader, device, regime_name="none", mix=None):
    """
    Train a regime and return history and the trained model.
    """
    mix_str = mix if mix else "none"
    logger.debug(f"Training regime: {regime_name} (mix={mix_str})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    loss_fn = nn.CrossEntropyLoss()

    history = fit(
        model, train_loader, val_loader, EPOCHS, optimizer, loss_fn, device, mix=mix
    )

    return history, model


def get_regime_history(regime_name, train_func):
    """
    Smart fallback: Load history if exists, otherwise train from scratch.
    Also saves the trained model for later evaluation.
    """
    history_path = {
        "none": AUGMENT_NONE_HISTORY,
        "std": AUGMENT_STD_HISTORY,
        "mix": AUGMENT_MIX_HISTORY,
    }[regime_name]

    model_path = {
        "none": AUGMENT_NONE_CHECKPOINT,
        "std": AUGMENT_STD_CHECKPOINT,
        "mix": AUGMENT_MIX_CHECKPOINT,
    }[regime_name]

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
    model = build_resnet18(num_classes=len(CLASSES), mode="finetune")
    model.load_state_dict(torch.load(BEST_RESNET_CHECKPOINT, map_location=device))
    model = model.to(device)
    logger.debug("Best model loaded successfully")
    return model


def load_regime_model(regime_name, device):
    """Load a regime-specific trained model."""
    model_path = {
        "none": AUGMENT_NONE_CHECKPOINT,
        "std": AUGMENT_STD_CHECKPOINT,
        "mix": AUGMENT_MIX_CHECKPOINT,
    }[regime_name]

    logger.debug(f"Loading {regime_name} model from {model_path}...")
    model = build_resnet18(num_classes=len(CLASSES), mode="finetune")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    logger.debug(f"{regime_name} model loaded successfully")
    return model


def main():
    logger.info("=" * 50)
    logger.info("B3: Augmentation Study")
    logger.info("=" * 50)

    start_time = time.time()

    set_seed(STUDENT_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.debug(f"Classes: {CLASSES}, batch_size={BATCH_SIZE}")

    # Load base data (ResNet: 224x224, ImageNet normalization)
    logger.info("Loading CIFAR-10 subset...")
    base_train_ds, val_ds, test_ds = get_cifar10_subset(
        root=DATA_ROOT,
        classes=CLASSES,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        image_size=224,
        imagenet_norm=True,
        seed=STUDENT_ID,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)

    # === Regime 1: No augmentation ===
    logger.info("\n=== B3: No Augmentation ===")
    train_loader_no_aug, val_loader_no_aug, test_loader = make_loaders(
        base_train_ds, val_ds, test_ds, batch_size=BATCH_SIZE
    )

    def train_none():
        model = load_base_model(device)
        return train_regime(
            model,
            train_loader_no_aug,
            val_loader_no_aug,
            device,
            regime_name="none",
            mix=None,
        )

    history_none, _model_none_path = get_regime_history("none", train_none)

    # === Regime 2: Standard augmentation (crop + flip) ===
    logger.info("\n=== B3: Standard Augmentation (Crop + Flip) ===")
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    augment_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(p=RANDOM_HORIZONTAL_FLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    full_train = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=augment_transform
    )

    def filter_by_class(dataset, class_list):
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_list]
        return Subset(dataset, indices)

    train_filtered = filter_by_class(full_train, CLASSES)

    rng = np.random.RandomState(STUDENT_ID)
    train_indices_by_class = {c: [] for c in CLASSES}
    for idx, (_, label) in enumerate(train_filtered):
        if label in CLASSES:
            train_indices_by_class[label].append(idx)

    train_selected = []
    for c in CLASSES:
        indices = train_indices_by_class[c]
        rng.shuffle(indices)
        train_selected.extend(indices[:N_TRAIN_PER_CLASS])

    val_size_per_class = 160
    train_size_per_class = N_TRAIN_PER_CLASS - val_size_per_class
    train_indices = []
    val_indices = []
    for c in CLASSES:
        class_indices = [idx for idx in train_selected if train_filtered[idx][1] == c]
        rng.shuffle(class_indices)
        train_indices.extend(class_indices[:train_size_per_class])
        val_indices.extend(class_indices[train_size_per_class:N_TRAIN_PER_CLASS])

    train_ds_aug = Subset(train_filtered, train_indices)
    val_ds_aug = Subset(train_filtered, val_indices)

    train_loader_std, val_loader_std, _ = make_loaders(
        train_ds_aug, val_ds_aug, test_ds, batch_size=BATCH_SIZE
    )

    def train_std():
        model = load_base_model(device)
        return train_regime(
            model, train_loader_std, val_loader_std, device, regime_name="std", mix=None
        )

    history_std, _model_std_path = get_regime_history("std", train_std)

    # === Regime 3: Mixup ===
    logger.info("\n=== B3: Mixup Augmentation ===")

    def train_mix():
        model = load_base_model(device)
        return train_regime(
            model,
            train_loader_std,
            val_loader_std,
            device,
            regime_name="mixup",
            mix="mixup",
        )

    history_mix, _model_mix_path = get_regime_history("mix", train_mix)

    # === Generate comparison figure ===
    plot_augment_comparison(history_none, history_std, history_mix, AUGMENT_COMPARE)

    # === Evaluate all regimes on test set ===
    logger.info("\n=== B3: Test Results ===")

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
    logger.info(f"B3 completed in {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
