"""
Tests for utility functions: accuracy, parameter counting, history I/O, plotting.
"""

import os
import tempfile

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # Use non-interactive backend for tests

from src.config import NUM_CLASSES
from src.models import SmallCNN
from src.utils import (
    accuracy,
    count_trainable_params,
    history_exists,
    load_history,
    plot_history,
    save_history,
)


class TestAccuracy:
    """Tests for accuracy function."""

    def test_accuracy_perfect(self):
        """Accuracy should be 1.0 when all predictions are correct."""
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        acc = accuracy(logits, targets)
        assert acc == 1.0

    def test_accuracy_zero(self):
        """Accuracy should be 0.0 when no predictions are correct."""
        logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        targets = torch.tensor([0, 1])
        acc = accuracy(logits, targets)
        assert acc == 0.0

    def test_accuracy_half(self):
        """Accuracy should be 0.5 when half predictions are correct."""
        logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        targets = torch.tensor([0, 1])
        acc = accuracy(logits, targets)
        assert acc == 0.5

    def test_accuracy_multiclass(self):
        """Accuracy should work with multiple classes."""
        logits = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        targets = torch.tensor([2, 0, 1, 0])
        acc = accuracy(logits, targets)
        # Expected: correct = [2, 0, 1, 2?] → actually [2, 0, 1, 2]
        # targets = [2, 0, 1, 0], so 3/4 correct
        assert acc == 0.75


class TestCountTrainableParams:
    """Tests for count_trainable_params function."""

    def test_smallcnn_parameters(self):
        """SmallCNN should have correct trainable parameters."""
        model = SmallCNN(num_classes=NUM_CLASSES)
        trainable = count_trainable_params(model)
        assert trainable == 94341

    def test_model_with_frozen_layers(self):
        """count_trainable_params should only count trainable layers."""
        model = SmallCNN(num_classes=NUM_CLASSES)

        # Freeze some layers
        for param in model.block1.parameters():
            param.requires_grad = False

        trainable = count_trainable_params(model)

        # Should be less than total
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total
        assert trainable > 0

    def test_empty_model(self):
        """Empty model should have 0 trainable parameters."""

        class EmptyModel(torch.nn.Module):
            pass

        model = EmptyModel()
        trainable = count_trainable_params(model)
        assert trainable == 0


class TestHistoryIO:
    """Tests for history save/load functions."""

    def test_save_and_load_history(self):
        """Saving and loading history should preserve data."""
        history = {
            "train_loss": [0.5, 0.3, 0.2],
            "train_acc": [0.6, 0.7, 0.8],
            "val_loss": [0.4, 0.25, 0.15],
            "val_acc": [0.65, 0.75, 0.85],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            save_history(history, filepath)
            loaded = load_history(filepath)

            assert loaded["train_loss"] == history["train_loss"]
            assert loaded["train_acc"] == history["train_acc"]
            assert loaded["val_loss"] == history["val_loss"]
            assert loaded["val_acc"] == history["val_acc"]
        finally:
            os.unlink(filepath)

    def test_history_exists(self):
        """history_exists should correctly detect existing files."""
        history = {
            "train_loss": [0.5, 0.3],
            "train_acc": [0.6, 0.7],
            "val_loss": [0.4, 0.25],
            "val_acc": [0.65, 0.75],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            save_history(history, filepath)
            assert history_exists(filepath) is True
            assert history_exists("nonexistent_file.csv") is False
        finally:
            os.unlink(filepath)

    def test_history_empty_file(self):
        """history_exists should return False for empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            assert history_exists(filepath) is False
        finally:
            os.unlink(filepath)

    def test_load_history_missing_file(self):
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_history("nonexistent_file.csv")

    def test_history_keys(self):
        """Loaded history should have all expected keys."""
        history = {
            "train_loss": [0.5, 0.3],
            "train_acc": [0.6, 0.7],
            "val_loss": [0.4, 0.25],
            "val_acc": [0.65, 0.75],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            save_history(history, filepath)
            loaded = load_history(filepath)

            expected_keys = {"train_loss", "train_acc", "val_loss", "val_acc"}
            assert set(loaded.keys()) == expected_keys
        finally:
            os.unlink(filepath)


class TestPlotting:
    """Tests for plotting functions (smoke tests)."""

    def test_plot_history_creates_file(self):
        """plot_history should create a PNG file."""
        history = {
            "train_loss": [0.5, 0.3, 0.2],
            "train_acc": [0.6, 0.7, 0.8],
            "val_loss": [0.4, 0.25, 0.15],
            "val_acc": [0.65, 0.75, 0.85],
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            plot_history(history, "Test Plot", filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_plot_history_with_different_epochs(self):
        """plot_history should handle different numbers of epochs."""
        history = {
            "train_loss": [0.5],
            "train_acc": [0.6],
            "val_loss": [0.4],
            "val_acc": [0.65],
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            plot_history(history, "Single Epoch", filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_plot_history_long(self):
        """plot_history should handle many epochs."""
        n_epochs = 50
        history = {
            "train_loss": [0.5 - i * 0.01 for i in range(n_epochs)],
            "train_acc": [0.5 + i * 0.01 for i in range(n_epochs)],
            "val_loss": [0.4 - i * 0.005 for i in range(n_epochs)],
            "val_acc": [0.55 + i * 0.005 for i in range(n_epochs)],
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            plot_history(history, "Long History", filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
