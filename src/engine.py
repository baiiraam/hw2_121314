# =====================================================================
# FILE: src/engine.py
# =====================================================================
"""
Training and evaluation engine for all experiments.
"""
import torch

from src.augment import cutmix_batch, mix_criterion, mixup_batch


def train_one_epoch(model, loader, optimizer, loss_fn, device, mix=None):
    """
    Train model for one epoch.

    Args:
        model: nn.Module to train
        loader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: torch.device
        mix: None, "mixup", or "cutmix"

    Returns:
        dict with "loss" and "acc" for the epoch

    Note on accuracy for mixed batches:
    For Mixup/CutMix, accuracy is computed against y_a (the first label
    in the mix). This is a simplification - the "true" label is a mix,
    but reporting accuracy against y_a gives a reasonable indication of
    model performance during training. The final evaluation is always
    done without mixing.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if mix == "mixup":
            x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=1.0)
            logits = model(x_mixed)
            loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)
            # Accuracy vs. dominant label y_a
            preds = logits.argmax(dim=1)
            correct += (preds == y_a).sum().item()

        elif mix == "cutmix":
            x_mixed, y_a, y_b, lam = cutmix_batch(x, y, alpha=1.0)
            logits = model(x_mixed)
            loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)
            # Accuracy vs. dominant label y_a
            preds = logits.argmax(dim=1)
            correct += (preds == y_a).sum().item()

        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total
    }


def evaluate(model, loader, loss_fn, device):
    """
    Evaluate model on a dataset.

    Returns:
        dict with "loss" and "acc"
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total
    }


def fit(model, train_loader, val_loader, epochs, optimizer, loss_fn,
        device, mix=None):
    """
    Full training loop with history tracking.

    Returns:
        dict with train_loss, train_acc, val_loss, val_acc lists
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, mix=mix
        )

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['acc']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['acc']:.4f}")

    return history