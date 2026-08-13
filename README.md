# HW2: CNNs and Computer Vision with PyTorch

Student ID: 121314

## Setup

requirements file for development:  **requirements-dev.txt**

requirements file for prod:  **requirements-prod.txt**

For assignment, it is recommended for reviewers to use prod, the only difference is that the prod requirements does not include linting, type checking, formatting, and testing.

## Run All Experiments

```bash
python run_all.py
```

This generates:
- `figures/cnn_curves.png` - SmallCNN training curves (B1)
- `figures/transfer_compare.png` - Transfer learning comparison (B2)
- `figures/augment_compare.png` - Augmentation study (B3)
- `figures/zoo_*.png` - Bonus inference results

## Individual Experiments

```bash
# B1: Train SmallCNN
python experiments/train_cnn.py

# B2: Transfer learning comparison
python experiments/transfer_compare.py

# B3: Augmentation study
python experiments/augment_compare.py

# Bonus: Inference on sample images
python experiments/bonus_inference.py
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**Results:** 70 tests passing, 77% coverage.

## Code Quality

```bash
# Linting
ruff check --fix .

# Type checking
ty check src/ tests/
```

All checks pass.

## Structure

```
hw2_121314/
├── src/
│   ├── data.py          # Data loading & reproducibility
│   ├── models.py        # SmallCNN, build_resnet18
│   ├── augment.py       # Mixup, CutMix, mix_criterion
│   ├── engine.py        # train_one_epoch, evaluate, fit
│   ├── utils.py         # accuracy, count_trainable_params, plot_*
│   └── config.py        # Centralized configuration
├── experiments/
│   ├── train_cnn.py     # B1
│   ├── transfer_compare.py  # B2
│   ├── augment_compare.py   # B3
│   └── bonus_inference.py   # Bonus
├── tests/               # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_augment.py
│   ├── test_engine.py
│   └── test_utils.py
├── models/              # Saved checkpoints (created at runtime)
├── figures/             # Generated plots (created at runtime)
├── data/                # CIFAR-10 and sample images
├── run_all.py
├── requirements.txt
├── pyproject.toml
├── conftest.py
└── README.md
```

## Key Features

- **Reproducible**: All randomness seeded with Student ID 121314
- **Model Persistence**: Best model saved from B2, loaded in B3
- **Self-implemented Augmentation**: Mixup and CutMix without external libs
- **Colab-ready**: Works on free T4 GPU runtime
- **Comprehensive Tests**: 70 tests with 77% coverage
- **Centralized Config**: All hyperparameters in `src/config.py`

## Results

| Model | Setting | Test Acc |
|-------|---------|----------|
| SmallCNN | from scratch | **70.30%** |
| ResNet-18 | feature extraction | **86.40%** |
| ResNet-18 | fine-tuning | **93.70%** |
| ResNet-18 (best) | No augmentation | **94.00%** |

## Bonus: Pretrained Model Inference

Run on 3 sample images (soccer players):
- **Detection**: Faster R-CNN with FPN → boxes with labels/scores
- **Segmentation**: Mask R-CNN with RoIAlign → instance masks
- **Pose**: Keypoint R-CNN → human skeletons

Outputs saved to `figures/zoo_*.png`

## Notes

- ImageNet normalization used for ResNet (Problem 6a)
- Smaller learning rate used for fine-tuning (Problem 6b)
- GAP reduces parameters from 10,245 to 645 (Part C1)
- Feature extraction: ~2,565 params, Fine-tuning: ~11.2M params (Part C2)