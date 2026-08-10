"""
Data loading and reproducibility utilities for HW2.
Student ID: 121314 (used as seed)
"""
import os
import random
import pickle
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from loguru import logger


def set_seed(seed: int) -> None:
    """Set all randomness sources for full reproducibility."""
    logger.debug(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set successfully")


def worker_init_fn(worker_id: int, seed: int = 121314) -> None:
    """Seed DataLoader workers for reproducibility when num_workers > 0."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_file_size(filepath: str) -> str:
    """Get human-readable file size."""
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def get_cached_cifar10_224(
    root: str,
    classes: list[int],
    n_train_per_class: int,
    n_test_per_class: int,
    imagenet_norm: bool,
    seed: int,
):
    """
    Load pre-resized CIFAR-10 at 224x224 from cache if available.
    If not, create and save the cache.
    """
    logger.debug("Entering get_cached_cifar10_224()")

    norm_str = "imagenet" if imagenet_norm else "cifar10"
    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cifar10_224_{norm_str}_seed{seed}.pkl")

    if os.path.exists(cache_file):
        logger.info(f"📂 Loading cached CIFAR-10 224x224 from {cache_file}")
        logger.debug(f"Cache file size: {get_file_size(cache_file)}")
        start = time.time()
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        elapsed = time.time() - start
        logger.info(f"✅ Cache loaded in {elapsed:.2f}s")
        logger.debug("Exiting get_cached_cifar10_224()")
        return data

    logger.info("🔨 Creating cache: resizing CIFAR-10 to 224x224...")
    start = time.time()

    # Create transform that resizes to 224x224 but DOES NOT normalize yet
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load full CIFAR-10 with resize only (no normalization)
    logger.debug("Loading CIFAR-10 dataset for cache creation")
    full_train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=resize_transform
    )
    full_test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=resize_transform
    )

    # Filter to specified classes
    def filter_by_class(dataset, class_list):
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_list]
        return Subset(dataset, indices)

    train_filtered = filter_by_class(full_train, classes)
    test_filtered = filter_by_class(full_test, classes)

    logger.debug(f"Filtered to classes {classes}: {len(train_filtered)} train, {len(test_filtered)} test")

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
        train_selected.extend(indices[:n_train_per_class])

    # Split train pool into train (640) and val (160) per class
    train_pool_by_class = {}
    for c in classes:
        class_indices = [idx for idx in train_selected
                         if train_filtered[idx][1] == c]
        rng.shuffle(class_indices)
        train_pool_by_class[c] = class_indices

    # Build train and val sets
    train_indices = []
    val_indices = []
    val_size_per_class = 160
    train_size_per_class = n_train_per_class - val_size_per_class

    for c in classes:
        pool = train_pool_by_class[c]
        train_indices.extend(pool[:train_size_per_class])
        val_indices.extend(pool[train_size_per_class:n_train_per_class])

    # Select test samples per class
    test_indices = []
    for c in classes:
        indices = test_class_indices[c]
        rng.shuffle(indices)
        test_indices.extend(indices[:n_test_per_class])

    # Create datasets (stored as lists of (tensor, label) for caching)
    train_data = [(train_filtered[idx][0], train_filtered[idx][1]) for idx in train_indices]
    val_data = [(train_filtered[idx][0], train_filtered[idx][1]) for idx in val_indices]
    test_data = [(test_filtered[idx][0], test_filtered[idx][1]) for idx in test_indices]

    # Cache to disk
    with open(cache_file, "wb") as f:
        pickle.dump((train_data, val_data, test_data), f)

    elapsed = time.time() - start
    logger.success(f"✅ Cache saved to {cache_file} in {elapsed:.2f}s")
    logger.debug(f"Cache file size: {get_file_size(cache_file)}")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    logger.info(f"Per class: {train_size_per_class} train, {val_size_per_class} val, {n_test_per_class} test")
    logger.debug("Exiting get_cached_cifar10_224()")
    return train_data, val_data, test_data


class CachedCIFAR10Dataset(Dataset):
    """Dataset wrapper for pre-cached CIFAR-10 data with on-the-fly normalization."""

    def __init__(self, data, normalize_transform):
        self.data = data
        self.normalize_transform = normalize_transform
        logger.debug(f"Created CachedCIFAR10Dataset with {len(data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_tensor, label = self.data[idx]
        img_tensor = self.normalize_transform(img_tensor)
        return img_tensor, label


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
    Uses caching for 224x224 images to avoid repeated resizing.
    """
    logger.debug("Entering get_cifar10_subset()")
    logger.debug(f"Parameters: image_size={image_size}, imagenet_norm={imagenet_norm}, seed={seed}")

    # ImageNet stats (used for ResNet - Problem 6a)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # CIFAR-10 stats (used for SmallCNN)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    # Choose normalization transform based on model type
    if imagenet_norm:
        mean, std = imagenet_mean, imagenet_std
        logger.debug("Using ImageNet normalization (ResNet)")
    else:
        mean, std = cifar10_mean, cifar10_std
        logger.debug("Using CIFAR-10 normalization (SmallCNN)")

    normalize_transform = transforms.Normalize(mean=mean, std=std)

    # For SmallCNN (image_size=32), don't use cache (small images, fast)
    if image_size == 32:
        logger.debug("Using direct loading (image_size=32, no cache)")
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_transform,
        ])

        full_train = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        full_test = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )

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
            train_selected.extend(indices[:n_train_per_class])

        # Split train pool into train (640) and val (160) per class
        train_pool_by_class = {}
        for c in classes:
            class_indices = [idx for idx in train_selected
                             if train_filtered[idx][1] == c]
            rng.shuffle(class_indices)
            train_pool_by_class[c] = class_indices

        # Build train and val sets
        train_indices = []
        val_indices = []
        val_size_per_class = 160
        train_size_per_class = n_train_per_class - val_size_per_class

        for c in classes:
            pool = train_pool_by_class[c]
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

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Per class: {train_size_per_class} train, {val_size_per_class} val, {n_test_per_class} test")
        logger.debug("Exiting get_cifar10_subset()")

        return train_dataset, val_dataset, test_dataset

    # For ResNet (image_size=224), use cached resized images
    else:
        logger.debug("Using cached 224x224 images")
        train_data, val_data, test_data = get_cached_cifar10_224(
            root, classes, n_train_per_class, n_test_per_class, imagenet_norm, seed
        )

        train_dataset = CachedCIFAR10Dataset(train_data, normalize_transform)
        val_dataset = CachedCIFAR10Dataset(val_data, normalize_transform)
        test_dataset = CachedCIFAR10Dataset(test_data, normalize_transform)

        logger.debug("Exiting get_cifar10_subset()")
        return train_dataset, val_dataset, test_dataset


def make_loaders(train_ds, val_ds, test_ds, batch_size: int):
    """
    Wrap datasets in DataLoaders with proper shuffling and worker seeding.
    """
    logger.debug("Entering make_loaders()")
    logger.debug(f"Batch size: {batch_size}")

    # On Windows, num_workers=0 avoids process spawning overhead
    num_workers = 0 if platform.system() == "Windows" else 2
    logger.debug(f"Using num_workers={num_workers} (platform: {platform.system()})")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    logger.debug(f"Created DataLoaders: train={len(train_loader)} batches, "
                 f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    logger.debug("Exiting make_loaders()")
    return train_loader, val_loader, test_loader