# HW2: CNNs and Computer Vision with PyTorch — Developer Documentation

**Student ID:** 121314
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Module Reference](#module-reference)
   - [src.config](#srcconfig)
   - [src.data](#srcdata)
   - [src.models](#srcmodels)
   - [src.augment](#srcaugment)
   - [src.engine](#srcengine)
   - [src.utils](#srcutils)
4. [CLI Reference](#cli-reference)
   - [run_all.py](#run_allpy)
   - [train_cnn.py](#train_cnnpy)
   - [transfer_compare.py](#transfer_comparepy)
   - [augment_compare.py](#augment_comparepy)
   - [bonus_inference.py](#bonus_inferencepy)
5. [Experiment Reference](#experiment-reference)
6. [Testing](#testing)
7. [Code Quality](#code-quality)
8. [Project Structure](#project-structure)

---

## Overview

This project implements a complete deep learning pipeline for image classification on a CIFAR-10 subset. It includes:

- **B1:** A custom CNN (`SmallCNN`) trained from scratch
- **B2:** Transfer learning with ResNet-18 (feature extraction vs fine-tuning)
- **B3:** Data augmentation study (None, Standard, Mixup)
- **Bonus:** Pretrained model inference (Detection, Segmentation, Pose)

All experiments are reproducible, seeded with Student ID `121314`, and include comprehensive logging and testing.

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hw2_121314.git
cd hw2_121314

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
uv pip install -r requirements-prod.txt

# Install development dependencies (for testing)
uv pip install -r requirements-dev.txt
```

### Dependencies

**Production (`requirements-prod.txt`):**

```txt
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
matplotlib==3.8.4
loguru==0.7.2
```

**Development (`requirements-dev.txt`):**

```txt
pytest==9.1.1
pytest-cov==7.1.0
ruff==0.5.0
ty==0.0.1-alpha.10
```

---

## Module Reference

### `src.config`

Centralized configuration for all experiments.

#### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `STUDENT_ID` | `int` | `121314` | Student ID (used as random seed) |
| `CLASSES` | `list[int]` | `[0,1,2,3,4]` | CIFAR-10 classes to use |
| `BATCH_SIZE` | `int` | `64` | Training batch size |
| `EPOCHS` | `int` | `20` | Number of training epochs |
| `LR_SMALLCNN` | `float` | `1e-3` | Learning rate for SmallCNN |
| `LR_FEATURE_EXTRACT` | `float` | `1e-3` | Learning rate for feature extraction |
| `LR_FINETUNE` | `float` | `1e-4` | Learning rate for fine-tuning (Problem 6b) |
| `MIXUP_ALPHA` | `float` | `1.0` | Beta distribution parameter for Mixup |
| `RANDOM_CROP_PADDING` | `int` | `8` | Padding for RandomCrop |
| `DATA_ROOT` | `str` | `"./data"` | Directory for CIFAR-10 and samples |
| `MODELS_DIR` | `str` | `"./models"` | Directory for checkpoints |
| `FIGURES_DIR` | `str` | `"./figures"` | Directory for generated figures |

#### Usage

```python
from src.config import STUDENT_ID, BATCH_SIZE, EPOCHS, MODELS_DIR

print(f"Running with seed: {STUDENT_ID}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Model directory: {MODELS_DIR}")
```

---

### `src.data`

Data loading and reproducibility utilities.

#### `set_seed(seed: int) -> None`

Sets all randomness sources for full reproducibility.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `seed` | `int` | ✅ | Random seed to set |

**What it controls:**
- Python `random` module
- NumPy random generator
- PyTorch CPU random
- PyTorch CUDA random (if available)
- CuDNN deterministic mode

**Example:**

```python
from src.data import set_seed

set_seed(121314)  # Fixes all randomness
```

---

#### `get_cifar10_subset(...) -> tuple[Dataset, Dataset, Dataset]`

Loads CIFAR-10, filters to specified classes, and splits into train/val/test.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `root` | `str` | ✅ | Directory to store/download CIFAR-10 |
| `classes` | `list[int]` | ✅ | Class indices to keep (0-9) |
| `n_train_per_class` | `int` | ✅ | Training samples per class |
| `n_test_per_class` | `int` | ✅ | Test samples per class |
| `image_size` | `int` | ✅ | 32 for SmallCNN, 224 for ResNet |
| `imagenet_norm` | `bool` | ✅ | Use ImageNet stats (ResNet) or CIFAR-10 stats |
| `seed` | `int` | ✅ | Random seed for reproducibility |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `Dataset` | Training dataset (640 samples/class) |
| 1 | `Dataset` | Validation dataset (160 samples/class) |
| 2 | `Dataset` | Test dataset (200 samples/class) |

**Example:**

```python
from src.data import get_cifar10_subset

# For SmallCNN: 32x32 with CIFAR-10 normalization
train_ds, val_ds, test_ds = get_cifar10_subset(
    root="./data",
    classes=[0, 1, 2, 3, 4],
    n_train_per_class=800,
    n_test_per_class=200,
    image_size=32,
    imagenet_norm=False,
    seed=121314,
)

# For ResNet: 224x224 with ImageNet normalization
train_ds, val_ds, test_ds = get_cifar10_subset(
    root="./data",
    classes=[0, 1, 2, 3, 4],
    n_train_per_class=800,
    n_test_per_class=200,
    image_size=224,
    imagenet_norm=True,
    seed=121314,
)
```

**Data Split:**

| Split | Per Class | Total | Shape (SmallCNN) | Shape (ResNet) |
|-------|-----------|-------|------------------|----------------|
| Train | 640 | 3,200 | (N,3,32,32) | (N,3,224,224) |
| Validation | 160 | 800 | (N,3,32,32) | (N,3,224,224) |
| Test | 200 | 1,000 | (N,3,32,32) | (N,3,224,224) |

---

#### `make_loaders(...) -> tuple[DataLoader, DataLoader, DataLoader]`

Wraps datasets in DataLoaders with proper shuffling and worker seeding.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `train_ds` | `Dataset` | ✅ | Training dataset |
| `val_ds` | `Dataset` | ✅ | Validation dataset |
| `test_ds` | `Dataset` | ✅ | Test dataset |
| `batch_size` | `int` | ✅ | Batch size for all loaders |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `DataLoader` | Training loader (shuffled) |
| 1 | `DataLoader` | Validation loader (not shuffled) |
| 2 | `DataLoader` | Test loader (not shuffled) |

**Example:**

```python
from src.data import make_loaders

train_loader, val_loader, test_loader = make_loaders(
    train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, batch_size=64
)
```

---

### `src.models`

Model definitions.

#### `SmallCNN(num_classes: int = 5)`

Custom CNN with 3 conv blocks, BatchNorm, ReLU, MaxPool, and GAP head.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_classes` | `int` | ❌ | Number of output classes (default: 5) |

**Architecture:**

| Block | Layer | Output Shape |
|-------|-------|--------------|
| Input | — | (N, 3, 32, 32) |
| Block 1 | Conv(3→32) → BN → ReLU → MaxPool(2) | (N, 32, 16, 16) |
| Block 2 | Conv(32→64) → BN → ReLU → MaxPool(2) | (N, 64, 8, 8) |
| Block 3 | Conv(64→128) → BN → ReLU → MaxPool(2) | (N, 128, 4, 4) |
| Head | GAP → Flatten → Linear(128→5) | (N, 5) |

**Parameter Count:** 94,341

**Example:**

```python
from src.models import SmallCNN

model = SmallCNN(num_classes=5)
x = torch.randn(4, 3, 32, 32)
logits = model(x)  # (4, 5)
```

---

#### `build_resnet18(num_classes: int, mode: str) -> nn.Module`

Builds pretrained ResNet-18 with a new head for transfer learning.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_classes` | `int` | ✅ | Number of output classes |
| `mode` | `str` | ✅ | `"feature_extract"` or `"finetune"` |

**Modes:**

| Mode | Backbone | Head | Trainable Params | LR |
|------|----------|------|------------------|----|
| `feature_extract` | Frozen | Trainable | 2,565 | 1e-3 |
| `finetune` | Trainable | Trainable | 11,179,077 | 1e-4 |

**Example:**

```python
from src.models import build_resnet18

# Feature extraction (backbone frozen)
model = build_resnet18(num_classes=5, mode="feature_extract")
# Trainable params: 2,565

# Fine-tuning (all layers trainable)
model = build_resnet18(num_classes=5, mode="finetune")
# Trainable params: 11,179,077
```

---

### `src.augment`

Custom implementations of Mixup and CutMix augmentation.

#### `mixup_batch(x: Tensor, y: Tensor, alpha: float = 1.0) -> tuple[Tensor, Tensor, Tensor, float]`

Creates convex combination of two images and labels.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `x` | `Tensor` | ✅ | Batch of images (N, C, H, W) |
| `y` | `Tensor` | ✅ | Batch of labels (N,) |
| `alpha` | `float` | ❌ | Beta distribution parameter (default: 1.0) |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `Tensor` | Mixed images (N, C, H, W) |
| 1 | `Tensor` | First set of labels (N,) |
| 2 | `Tensor` | Second set of labels (N,) |
| 3 | `float` | Mixing coefficient (λ) |

**Example:**

```python
from src.augment import mixup_batch

x = torch.randn(64, 3, 32, 32)
y = torch.randint(0, 5, (64,))

x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=1.0)
# x_mixed = lam * x + (1 - lam) * x[idx]
# Loss = lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)
```

---

#### `cutmix_batch(x: Tensor, y: Tensor, alpha: float = 1.0) -> tuple[Tensor, Tensor, Tensor, float]`

Pastes a random box from one image onto another.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `x` | `Tensor` | ✅ | Batch of images (N, C, H, W) |
| `y` | `Tensor` | ✅ | Batch of labels (N,) |
| `alpha` | `float` | ❌ | Beta distribution parameter (default: 1.0) |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `Tensor` | Mixed images (N, C, H, W) |
| 1 | `Tensor` | First set of labels (N,) |
| 2 | `Tensor` | Second set of labels (N,) |
| 3 | `float` | Mixing coefficient (λ = 1 - box_area/total_area) |

**Edge Cases:** Box dimensions are clamped to `[1, W-1]` to avoid `randint` crashes.

**Example:**

```python
from src.augment import cutmix_batch

x = torch.randn(64, 3, 32, 32)
y = torch.randint(0, 5, (64,))

x_mixed, y_a, y_b, lam = cutmix_batch(x, y, alpha=1.0)
```

---

#### `mix_criterion(loss_fn: Callable, logits: Tensor, y_a: Tensor, y_b: Tensor, lam: float) -> Tensor`

Computes convex combination of losses for Mixup/CutMix.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `loss_fn` | `Callable` | ✅ | Loss function (e.g., CrossEntropyLoss) |
| `logits` | `Tensor` | ✅ | Model outputs (N, C) |
| `y_a` | `Tensor` | ✅ | First set of labels (N,) |
| `y_b` | `Tensor` | ✅ | Second set of labels (N,) |
| `lam` | `float` | ✅ | Mixing coefficient |

**Returns:**

| Type | Description |
|------|-------------|
| `Tensor` | Scalar loss value |

**Example:**

```python
from src.augment import mix_criterion

loss_fn = nn.CrossEntropyLoss()
loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)
# loss = lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)
```

---

### `src.engine`

Training and evaluation engine.

#### `train_one_epoch(...) -> dict`

Trains model for one epoch with optional Mixup/CutMix.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `nn.Module` | ✅ | Model to train |
| `loader` | `DataLoader` | ✅ | Training data loader |
| `optimizer` | `Optimizer` | ✅ | Optimizer for updating weights |
| `loss_fn` | `Callable` | ✅ | Loss function |
| `device` | `torch.device` | ✅ | CPU or CUDA device |
| `mix` | `str | None` | ❌ | `"mixup"`, `"cutmix"`, or `None` |

**Returns:**

```python
{
    "loss": 0.5,  # Average loss for the epoch
    "acc": 0.8,  # Average accuracy for the epoch
}
```

**Note:** For Mixup/CutMix, accuracy is computed against `y_a` (the dominant label).

**Example:**

```python
from src.engine import train_one_epoch

metrics = train_one_epoch(
    model=model,
    loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    mix="mixup",
)
```

---

#### `evaluate(...) -> dict`

Evaluates model on a dataset (no gradients).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `nn.Module` | ✅ | Model to evaluate |
| `loader` | `DataLoader` | ✅ | Data loader for evaluation |
| `loss_fn` | `Callable` | ✅ | Loss function |
| `device` | `torch.device` | ✅ | CPU or CUDA device |

**Returns:**

```python
{
    "loss": 0.4,  # Average loss
    "acc": 0.85,  # Average accuracy
}
```

**Example:**

```python
from src.engine import evaluate

metrics = evaluate(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
```

---

#### `fit(...) -> dict`

Full training loop with history tracking.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `nn.Module` | ✅ | Model to train |
| `train_loader` | `DataLoader` | ✅ | Training data loader |
| `val_loader` | `DataLoader` | ✅ | Validation data loader |
| `epochs` | `int` | ✅ | Number of training epochs |
| `optimizer` | `Optimizer` | ✅ | Optimizer |
| `loss_fn` | `Callable` | ✅ | Loss function |
| `device` | `torch.device` | ✅ | CPU or CUDA device |
| `mix` | `str | None` | ❌ | `"mixup"`, `"cutmix"`, or `None` |

**Returns:**

```python
{
    "train_loss": [0.5, 0.3, 0.2, ...],
    "train_acc": [0.6, 0.7, 0.8, ...],
    "val_loss": [0.4, 0.25, 0.15, ...],
    "val_acc": [0.65, 0.75, 0.85, ...],
}
```

**Example:**

```python
from src.engine import fit

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    mix="mixup",
)
```

---

### `src.utils`

Utility functions.

#### `accuracy(logits: Tensor, targets: Tensor) -> float`

Computes classification accuracy.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `logits` | `Tensor` | ✅ | Model outputs (N, C) |
| `targets` | `Tensor` | ✅ | Ground truth labels (N,) |

**Returns:** `float` — Accuracy in [0, 1]

**Example:**

```python
from src.utils import accuracy

acc = accuracy(logits, targets)
```

---

#### `count_trainable_params(model: nn.Module) -> int`

Counts parameters with `requires_grad=True`.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `nn.Module` | ✅ | Model to inspect |

**Returns:** `int` — Number of trainable parameters

**Example:**

```python
from src.utils import count_trainable_params

trainable = count_trainable_params(model)
print(f"Trainable params: {trainable:,}")
```

---

#### `save_history(history: dict, filepath: str) -> None`

Saves training history to CSV.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `history` | `dict` | ✅ | History dictionary from `fit()` |
| `filepath` | `str` | ✅ | Path to save CSV |

**CSV Format:** `epoch, train_loss, train_acc, val_loss, val_acc`

**Example:**

```python
from src.utils import save_history

save_history(history, "models/history.csv")
```

---

#### `load_history(filepath: str) -> dict`

Loads training history from CSV.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filepath` | `str` | ✅ | Path to CSV file |

**Returns:** `dict` — History dictionary

**Example:**

```python
from src.utils import load_history

history = load_history("models/history.csv")
```

---

#### `plot_history(history: dict, title: str, save_path: str) -> None`

Generates training curves (loss and accuracy) and saves to file.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `history` | `dict` | ✅ | History dictionary from `fit()` |
| `title` | `str` | ✅ | Plot title |
| `save_path` | `str` | ✅ | Path to save PNG |

**Example:**

```python
from src.utils import plot_history

plot_history(history, "Training Curves", "figures/curves.png")
```

---

#### `plot_transfer_comparison(...) -> None`

Plots validation accuracy for SmallCNN, feature extraction, and fine-tuning.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cnn_history` | `dict` | ✅ | SmallCNN history |
| `fe_history` | `dict` | ✅ | Feature extraction history |
| `ft_history` | `dict` | ✅ | Fine-tuning history |
| `save_path` | `str` | ✅ | Path to save PNG |

**Example:**

```python
from src.utils import plot_transfer_comparison

plot_transfer_comparison(
    cnn_history, fe_history, ft_history, "figures/transfer_compare.png"
)
```

---

#### `plot_augment_comparison(...) -> None`

Plots validation accuracy for three augmentation regimes.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `none_history` | `dict` | ✅ | No augmentation history |
| `std_history` | `dict` | ✅ | Standard augmentation history |
| `mix_history` | `dict` | ✅ | Mixup history |
| `save_path` | `str` | ✅ | Path to save PNG |

**Example:**

```python
from src.utils import plot_augment_comparison

plot_augment_comparison(
    none_history, std_history, mix_history, "figures/augment_compare.png"
)
```

---

## CLI Reference

### `run_all.py`

Orchestrates all experiments (B1, B2, B3, Bonus) in a single command.

**Usage:**

```bash
python run_all.py [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--verbose` | Show all output (INFO level — default) |
| `--quiet` | Suppress detailed output (WARNING level) |
| `--debug` | Show debug output (DEBUG level) |

**Examples:**

```bash
# Run all experiments with default verbosity
python run_all.py

# Run with debug output
python run_all.py --debug

# Run quietly (only warnings and errors)
python run_all.py --quiet
```

**Outputs:**

| File | Description |
|------|-------------|
| `figures/cnn_curves.png` | SmallCNN training curves (B1) |
| `figures/transfer_compare.png` | Transfer learning comparison (B2) |
| `figures/augment_compare.png` | Augmentation study (B3) |
| `figures/zoo_*.png` | Bonus inference results |

---

### `train_cnn.py` — B1: SmallCNN Baseline

**Usage:**

```bash
python experiments/train_cnn.py
```

**What it does:**

1. Loads CIFAR-10 subset (32×32, CIFAR-10 normalization)
2. Creates `SmallCNN` model
3. Trains for 20 epochs using Adam (`lr=1e-3`)
4. Saves model to `models/smallcnn.pth`
5. Saves history to `models/smallcnn_history.csv`
6. Generates `figures/cnn_curves.png`
7. Reports test accuracy

**Outputs:**

| File | Description |
|------|-------------|
| `models/smallcnn.pth` | Trained model weights |
| `models/smallcnn_history.csv` | Training history (20 epochs) |
| `figures/cnn_curves.png` | Loss and accuracy curves |

**Example Output:**

```
Using device: cuda
Train: 3200, Val: 800, Test: 1000
Epoch 1/20: Train Loss: 1.1938, Train Acc: 0.5294, Val Loss: 1.0503, Val Acc: 0.5850
...
Test Accuracy: 0.7030
✅ Model saved to models/smallcnn.pth
✅ History saved to models/smallcnn_history.csv
✅ Figure saved to figures/cnn_curves.png
```

---

### `transfer_compare.py` — B2: Transfer Learning Comparison

**Usage:**

```bash
python experiments/transfer_compare.py
```

**What it does:**

1. Loads CIFAR-10 subset (224×224, ImageNet normalization)
2. Creates two ResNet-18 models:
   - `feature_extract`: Backbone frozen, only new head trains (2,565 params)
   - `finetune`: All layers trainable (11,179,077 params, `lr=1e-4`)
3. Trains both for 20 epochs
4. Loads SmallCNN history for comparison
5. Generates `figures/transfer_compare.png`
6. Saves best fine-tuned model to `models/best_resnet_ft.pth`
7. Reports parameter counts and test accuracies

**Outputs:**

| File | Description |
|------|-------------|
| `models/feature_extract_model.pth` | Feature extraction model weights |
| `models/feature_extract_history.csv` | Feature extraction history |
| `models/finetune_model.pth` | Fine-tuning model weights |
| `models/finetune_history.csv` | Fine-tuning history |
| `models/best_resnet_ft.pth` | Best model (for B3) |
| `figures/transfer_compare.png` | Three validation curves |

**Example Output:**

```
Using device: cuda
Trainable params (feature_extract): 2,565
Test Accuracy: 0.8640
Trainable params (finetune): 11,179,077
Test Accuracy: 0.9370
✅ Figure saved to figures/transfer_compare.png
✅ Best model saved to models/best_resnet_ft.pth
```

---

### `augment_compare.py` — B3: Augmentation Study

**Usage:**

```bash
python experiments/augment_compare.py
```

**What it does:**

1. Loads CIFAR-10 subset (224×224, ImageNet normalization)
2. Loads best ResNet-18 from B2 (`models/best_resnet_ft.pth`)
3. Trains under three regimes:
   - **None:** No augmentation
   - **Standard:** RandomCrop(224, padding=8) + RandomHorizontalFlip
   - **Mixup:** Convex combinations of images and labels (`alpha=1.0`)
4. Saves each regime's model and history
5. Generates `figures/augment_compare.png`
6. Reports test accuracies for all three regimes

**Outputs:**

| File | Description |
|------|-------------|
| `models/augment_none.pth` | None regime model |
| `models/none_history.csv` | None regime history |
| `models/augment_std.pth` | Standard regime model |
| `models/std_history.csv` | Standard regime history |
| `models/augment_mix.pth` | Mixup regime model |
| `models/mix_history.csv` | Mixup regime history |
| `figures/augment_compare.png` | Three validation curves |

**Example Output:**

```
Using device: cuda
=== B3: No Augmentation ===
Test Acc: 0.9400
=== B3: Standard Augmentation ===
Test Acc: 0.9170
=== B3: Mixup ===
Test Acc: 0.9340
✅ Figure saved to figures/augment_compare.png
```

---

### `bonus_inference.py` — Bonus: Pretrained Model Inference

**Usage:**

```bash
python experiments/bonus_inference.py
```

**Prerequisites:** Place 3-5 images in `data/samples/` (JPG, JPEG, PNG).

**What it does:**

1. Scans `data/samples/` for images
2. For each image, runs three models:
   - **Detection:** `fasterrcnn_resnet50_fpn` → boxes with labels/scores
   - **Segmentation:** `maskrcnn_resnet50_fpn` → instance masks
   - **Pose:** `keypointrcnn_resnet50_fpn` → human skeletons
3. Saves annotated outputs to `figures/zoo_*.png`

**Outputs:**

| File Pattern | Description |
|--------------|-------------|
| `figures/zoo_detection_*.png` | Bounding boxes |
| `figures/zoo_segmentation_*.png` | Instance masks |
| `figures/zoo_pose_*.png` | Human skeletons |

**Example Output:**

```
Using device: cuda
Found 3 sample images
Running detection on: player.jpg
Found 2 detections above threshold 0.5
✅ Detection result saved to figures/zoo_detection_player.png
Running segmentation on: player.jpg
Found 2 segmentations above threshold 0.5
✅ Segmentation result saved to figures/zoo_segmentation_player.png
Running pose estimation on: player.jpg
Found 1 person(s) with keypoints above threshold 0.5
✅ Pose result saved to figures/zoo_pose_player.png
```

---

## Experiment Reference

| Experiment | Script | Purpose | Test Accuracy |
|------------|--------|---------|---------------|
| **B1** | `train_cnn.py` | SmallCNN from scratch | 70.30% |
| **B2** | `transfer_compare.py` | Feature extraction | 86.40% |
| **B2** | `transfer_compare.py` | Fine-tuning | 93.70% |
| **B3** | `augment_compare.py` | No augmentation | 94.00% |
| **B3** | `augment_compare.py` | Standard augmentation | 91.70% |
| **B3** | `augment_compare.py` | Mixup | 93.40% |
| **Bonus** | `bonus_inference.py` | Detection/Segmentation/Pose | — |

---

## Testing

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
uv run python -m pytest tests/test_models.py -v
```

### Test Coverage

| File | Coverage | Status |
|------|----------|--------|
| `src/augment.py` | 100% | ✅ |
| `src/config.py` | 100% | ✅ |
| `src/engine.py` | 99% | ✅ |
| `src/models.py` | 100% | ✅ |
| `src/data.py` | 68% | ⚠️ |
| `src/utils.py` | 66% | ⚠️ |
| `src/log_config.py` | 0% | ⚠️ |
| **Total** | **77%** | ✅ |

### Test Files

| File | Tests | What It Tests |
|------|-------|---------------|
| `test_data.py` | 10 | Data loading, reproducibility, transforms |
| `test_models.py` | 15 | SmallCNN, ResNet-18, parameter counts |
| `test_augment.py` | 16 | Mixup, CutMix, loss criterion |
| `test_engine.py` | 14 | Training loops, evaluation, fit |
| `test_utils.py` | 15 | Accuracy, history I/O, plotting |

---

## Code Quality

### Linting

```bash
uv run ruff check --fix .
```

### Type Checking

```bash
uv run ty check src/ tests/
```

All checks pass.

---

## Project Structure

```
hw2_121314/
├── src/
│   ├── data.py          # Data loading & reproducibility
│   ├── models.py        # SmallCNN, build_resnet18
│   ├── augment.py       # Mixup, CutMix, mix_criterion
│   ├── engine.py        # train_one_epoch, evaluate, fit
│   ├── utils.py         # accuracy, count_trainable_params, plot_*
│   ├── config.py        # Centralized configuration
│   └── log_config.py    # Logging setup
├── experiments/
│   ├── train_cnn.py     # B1: SmallCNN baseline
│   ├── transfer_compare.py  # B2: Transfer learning comparison
│   ├── augment_compare.py   # B3: Augmentation study
│   └── bonus_inference.py   # Bonus: Pretrained model inference
├── tests/               # Unit tests
│   ├── conftest.py      # pytest configuration
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_augment.py
│   ├── test_engine.py
│   └── test_utils.py
├── models/              # Saved checkpoints (created at runtime)
├── figures/             # Generated plots (created at runtime)
├── data/                # CIFAR-10 and sample images
├── run_all.py
├── requirements-prod.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## Key Features

- **Reproducible:** All randomness seeded with Student ID 121314
- **Model Persistence:** Best model saved from B2, loaded in B3
- **Self-implemented Augmentation:** Mixup and CutMix without external libs
- **Colab-ready:** Works on free T4 GPU runtime
- **Comprehensive Tests:** 70 tests with 77% coverage
- **Centralized Config:** All hyperparameters in `src/config.py`
- **Smart Fallback:** Caches histories and models to avoid retraining
- **Logging:** Structured logging with `loguru` (console + file)

---

## Notes

- ImageNet normalization used for ResNet (Problem 6a)
- Smaller learning rate used for fine-tuning (Problem 6b)
- GAP reduces parameters from 10,245 to 645 (Part C1)
- Feature extraction: ~2,565 params, Fine-tuning: ~11.2M params (Part C2)
- CutMix edge cases are guarded against (box dimensions clamped)
- On Windows, `num_workers=0` is automatically used to avoid process spawning overhead