"""
Data loading and reproducibility utilities for HW2.
Student ID: 121314 (used as seed)
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    """Set all randomness sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, seed: int = 121314) -> None:
    """Seed DataLoader workers for reproducibility when num_workers > 0."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cifar10_subset(
    root: str,
    classes: list[int],
    n_train_per_class: int,
    n_test_per_class: int,
    image_size: int,
    imagenet_norm: bool,
    seed: int,
):
    """
    Load CIFAR-10, filter to specified classes, split into train/val/test.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # CIFAR-10 mean/std for normalization (used when imagenet_norm=False)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    # ImageNet stats (used for ResNet - Problem 6a)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Choose transforms based on model type
    if imagenet_norm:
        # For ResNet: resize to 224x224, use ImageNet stats
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # For SmallCNN: keep 32x32, use CIFAR-10 stats
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ])

    # Download full CIFAR-10
    full_train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    full_test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    # Filter to specified classes
    def filter_by_class(dataset, class_list):
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_list]
        return Subset(dataset, indices)

    train_filtered = filter_by_class(full_train, classes)
    test_filtered = filter_by_class(full_test, classes)

    # Group indices by class for stratified sampling
    def group_by_class(subset):
        class_to_indices = {c: [] for c in classes}
        for idx, (_, label) in enumerate(subset):
            class_to_indices[label].append(idx)
        return class_to_indices

    train_class_indices = group_by_class(train_filtered)
    test_class_indices = group_by_class(test_filtered)

    # Use seed for reproducible shuffling
    rng = np.random.RandomState(seed)

    # Select train samples per class
    train_selected = []
    for c in classes:
        indices = train_class_indices[c]
        rng.shuffle(indices)
        # Take n_train_per_class for the train pool
        train_selected.extend(indices[:n_train_per_class])

    # Split train pool into train (640) and val (160) per class
    # Create separate pools per class for stratified split
    train_pool_by_class = {}
    for c in classes:
        class_indices = [idx for idx in train_selected
                         if train_filtered[idx][1] == c]
        # Shuffle using seed for reproducibility
        rng.shuffle(class_indices)
        train_pool_by_class[c] = class_indices

    # Build train and val sets
    train_indices = []
    val_indices = []
    val_size_per_class = 160  # 20% of 800
    train_size_per_class = n_train_per_class - val_size_per_class

    for c in classes:
        pool = train_pool_by_class[c]
        # Split with seeded random_split equivalent
        train_indices.extend(pool[:train_size_per_class])
        val_indices.extend(pool[train_size_per_class:n_train_per_class])

    # Select test samples per class
    test_indices = []
    for c in classes:
        indices = test_class_indices[c]
        rng.shuffle(indices)
        test_indices.extend(indices[:n_test_per_class])

    # Create datasets
    train_dataset = Subset(train_filtered, train_indices)
    val_dataset = Subset(train_filtered, val_indices)
    test_dataset = Subset(test_filtered, test_indices)

    # Verify counts
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}")
    print(f"Per class: {train_size_per_class} train, "
          f"{val_size_per_class} val, {n_test_per_class} test")

    return train_dataset, val_dataset, test_dataset


def make_loaders(train_ds, val_ds, test_ds, batch_size: int):
    """
    Wrap datasets in DataLoaders with proper shuffling and worker seeding.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader