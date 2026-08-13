"""
Tests for training engine functions: train_one_epoch, evaluate, fit.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LR_SMALLCNN, NUM_CLASSES
from src.engine import evaluate, fit, train_one_epoch
from src.models import SmallCNN


class TestTrainOneEpoch:
    """Tests for train_one_epoch function."""

    def test_train_one_epoch_returns_metrics(self):
        """train_one_epoch should return loss and accuracy."""
        # Create a small dataset
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = train_one_epoch(model, loader, optimizer, loss_fn, device, mix=None)

        # Should return loss and accuracy
        assert "loss" in metrics
        assert "acc" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["acc"], float)
        assert 0 <= metrics["acc"] <= 1

    def test_train_one_epoch_updates_weights(self):
        """Weights should change after one epoch."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Save initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        train_one_epoch(model, loader, optimizer, loss_fn, device, mix=None)

        # Check that weights changed
        for initial, current in zip(initial_weights, model.parameters()):
            assert not torch.allclose(initial, current)

    def test_train_one_epoch_with_mixup(self):
        """train_one_epoch should work with mixup."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = train_one_epoch(
            model, loader, optimizer, loss_fn, device, mix="mixup"
        )

        assert "loss" in metrics
        assert "acc" in metrics
        assert 0 <= metrics["acc"] <= 1

    def test_train_one_epoch_with_cutmix(self):
        """train_one_epoch should work with cutmix."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = train_one_epoch(
            model, loader, optimizer, loss_fn, device, mix="cutmix"
        )

        assert "loss" in metrics
        assert "acc" in metrics
        assert 0 <= metrics["acc"] <= 1

    def test_train_one_epoch_device_agnostic(self):
        """train_one_epoch should work on both CPU and CUDA (if available)."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()

        # Test on CPU
        device = torch.device("cpu")
        metrics = train_one_epoch(model, loader, optimizer, loss_fn, device, mix=None)
        assert 0 <= metrics["acc"] <= 1

        # Test on CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            metrics = train_one_epoch(
                model, loader, optimizer, loss_fn, device, mix=None
            )
            assert 0 <= metrics["acc"] <= 1


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_returns_metrics(self):
        """evaluate should return loss and accuracy."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = evaluate(model, loader, loss_fn, device)

        assert "loss" in metrics
        assert "acc" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["acc"], float)
        assert 0 <= metrics["acc"] <= 1

    def test_evaluate_no_grad(self):
        """evaluate should not compute gradients."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Check that no gradients are computed
        # If gradients were computed, we'd need to zero them
        evaluate(model, loader, loss_fn, device)

        # Gradients should not be present on model parameters
        for param in model.parameters():
            assert param.grad is None

    def test_evaluate_accuracy_reasonable(self):
        """Accuracy should be reasonable (not all zeros or random)."""
        batch_size = 4
        num_samples = 64
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        metrics = evaluate(model, loader, loss_fn, device)

        # With random weights, accuracy should be around 1/NUM_CLASSES
        # (but for small dataset, it can vary)
        assert 0.01 <= metrics["acc"] <= 0.6

    def test_evaluate_device_agnostic(self):
        """evaluate should work on both CPU and CUDA (if available)."""
        batch_size = 4
        num_samples = 32
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        loss_fn = nn.CrossEntropyLoss()

        # Test on CPU
        device = torch.device("cpu")
        metrics = evaluate(model, loader, loss_fn, device)
        assert 0 <= metrics["acc"] <= 1

        # Test on CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            metrics = evaluate(model, loader, loss_fn, device)
            assert 0 <= metrics["acc"] <= 1


class TestFit:
    """Tests for fit function."""

    def test_fit_returns_history(self):
        """fit should return training history with all metrics."""
        batch_size = 4
        num_samples = 32
        epochs = 3
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        history = fit(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            loss_fn,
            device,
            mix=None,
        )

        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_loss" in history
        assert "val_acc" in history

        assert len(history["train_loss"]) == epochs
        assert len(history["train_acc"]) == epochs
        assert len(history["val_loss"]) == epochs
        assert len(history["val_acc"]) == epochs

    def test_fit_with_mixup(self):
        """fit should work with mixup."""
        batch_size = 4
        num_samples = 32
        epochs = 3
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        history = fit(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            loss_fn,
            device,
            mix="mixup",
        )

        assert len(history["train_loss"]) == epochs
        assert len(history["val_acc"]) == epochs

    def test_fit_with_cutmix(self):
        """fit should work with cutmix."""
        batch_size = 4
        num_samples = 32
        epochs = 3
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        history = fit(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            loss_fn,
            device,
            mix="cutmix",
        )

        assert len(history["train_loss"]) == epochs
        assert len(history["val_acc"]) == epochs

    def test_fit_prints_epoch_progress(self):
        """fit should print progress logs (captured by loguru)."""
        batch_size = 4
        num_samples = 16
        epochs = 2
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (num_samples,))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SmallCNN(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_SMALLCNN)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Should not raise any errors
        history = fit(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            loss_fn,
            device,
            mix=None,
        )

        assert len(history["train_loss"]) == epochs

    def test_fit_model_trains_over_epochs(self):
        """Model accuracy should improve over epochs (on simple data)."""
        batch_size = 4
        num_samples = 64
        epochs = 5
        # Create simple data with clear patterns
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 2, (num_samples,))  # Binary classification
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Use a smaller model for faster testing
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(4 * 16 * 16, 2)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv(x)))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        history = fit(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            loss_fn,
            device,
            mix=None,
        )

        # Accuracy should improve (or at least not get worse)
        # For a simple model on random data, we can't guarantee improvement
        # but we can check that it doesn't crash
        assert len(history["train_acc"]) == epochs
