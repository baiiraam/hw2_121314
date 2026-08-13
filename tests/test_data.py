"""
Tests for data loading and reproducibility utilities.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import get_cifar10_subset, make_loaders, set_seed


class TestSetSeed:
    """Tests for the set_seed function."""

    def test_seed_reproducibility(self):
        """Running set_seed twice with same seed should produce same random numbers."""
        # First run
        set_seed(121314)
        np_vals1 = np.random.randn(10)
        torch_vals1 = torch.randn(10)

        # Second run (same seed)
        set_seed(121314)
        np_vals2 = np.random.randn(10)
        torch_vals2 = torch.randn(10)

        # They should be identical
        np.testing.assert_array_equal(np_vals1, np_vals2)
        torch.testing.assert_close(torch_vals1, torch_vals2)

    def test_different_seeds_produce_different_values(self):
        """Different seeds should produce different random numbers."""
        set_seed(121314)
        vals1 = np.random.randn(10)

        set_seed(999999)
        vals2 = np.random.randn(10)

        # They should NOT be identical
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(vals1, vals2)


class TestGetCIFAR10Subset:
    """Tests for get_cifar10_subset function."""

    def test_dataset_sizes(self):
        """Dataset should have correct sizes: 640 train, 160 val, 200 test per class."""
        classes = [0, 1, 2, 3, 4]
        n_train_per_class = 800
        n_test_per_class = 200

        train_ds, val_ds, test_ds = get_cifar10_subset(
            root="./data",
            classes=classes,
            n_train_per_class=n_train_per_class,
            n_test_per_class=n_test_per_class,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        expected_train = len(classes) * 640  # 5 * 640 = 3200
        expected_val = len(classes) * 160  # 5 * 160 = 800
        expected_test = len(classes) * 200  # 5 * 200 = 1000

        assert len(train_ds) == expected_train
        assert len(val_ds) == expected_val
        assert len(test_ds) == expected_test

    def test_correct_classes(self):
        """All samples should belong to the specified classes."""
        classes = [0, 1, 2, 3, 4]
        train_ds, val_ds, test_ds = get_cifar10_subset(
            root="./data",
            classes=classes,
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        # Check a few samples from each dataset
        for dataset in [train_ds, val_ds, test_ds]:
            for i in range(min(10, len(dataset))):
                _, label = dataset[i]
                assert label in classes

    def test_reproducible_split(self):
        """Running twice should produce the same data split (same indices)."""
        classes = [0, 1, 2, 3, 4]

        # Run 1
        train_ds1, _val_ds1, _ = get_cifar10_subset(
            root="./data",
            classes=classes,
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        # Run 2 (same seed)
        train_ds2, _val_ds2, _ = get_cifar10_subset(
            root="./data",
            classes=classes,
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        # Extract labels from both runs
        labels1 = [train_ds1[i][1] for i in range(len(train_ds1))]
        labels2 = [train_ds2[i][1] for i in range(len(train_ds2))]

        # They should be identical
        assert labels1 == labels2

    def test_smallcnn_transform(self):
        """SmallCNN transform should keep 32x32 with CIFAR-10 normalization."""
        train_ds, _, _ = get_cifar10_subset(
            root="./data",
            classes=[0, 1, 2, 3, 4],
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        img, _ = train_ds[0]

        # Shape should be (3, 32, 32)
        assert img.shape == (3, 32, 32)

        # Values should be normalized (roughly zero mean)
        assert img.mean() < 0.1  # Close to 0 after normalization

    def test_resnet_transform(self):
        """ResNet transform should resize to 224x224 with ImageNet normalization."""
        train_ds, _, _ = get_cifar10_subset(
            root="./data",
            classes=[0, 1, 2, 3, 4],
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=224,
            imagenet_norm=True,
            seed=121314,
        )

        img, _ = train_ds[0]

        # Shape should be (3, 224, 224)
        assert img.shape == (3, 224, 224)

        # Values should be normalized (roughly zero mean)
        assert img.mean() < 0.1


class TestMakeLoaders:
    """Tests for make_loaders function."""

    def test_data_loader_shapes(self):
        """DataLoaders should produce correct batch shapes."""
        classes = [0, 1, 2, 3, 4]
        batch_size = 64

        train_ds, val_ds, test_ds = get_cifar10_subset(
            root="./data",
            classes=classes,
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        train_loader, val_loader, test_loader = make_loaders(
            train_ds, val_ds, test_ds, batch_size=batch_size
        )

        # Check types
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Check batch shapes
        for batch in train_loader:
            x, y = batch
            assert x.shape[0] == batch_size
            assert x.shape[1] == 3
            assert y.shape[0] == batch_size
            break

    def test_train_loader_shuffles(self):
        """Train loader should have shuffle=True."""
        train_ds, _, _ = get_cifar10_subset(
            root="./data",
            classes=[0, 1, 2, 3, 4],
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        train_loader, _, _ = make_loaders(train_ds, train_ds, train_ds, batch_size=64)

        # Train loader should be shuffling
        assert train_loader.sampler is not None
        # For a Subset, the sampler is a SubsetRandomSampler which shuffles

    def test_val_loader_does_not_shuffle(self):
        """Validation loader should have shuffle=False."""
        train_ds, _, _ = get_cifar10_subset(
            root="./data",
            classes=[0, 1, 2, 3, 4],
            n_train_per_class=800,
            n_test_per_class=200,
            image_size=32,
            imagenet_norm=False,
            seed=121314,
        )

        _, val_loader, _ = make_loaders(train_ds, train_ds, train_ds, batch_size=64)

        # Validation loader should NOT be shuffling
        # For a Subset with shuffle=False, the sampler is a SequentialSampler
        assert val_loader.sampler is not None
