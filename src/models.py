"""
Model definitions: SmallCNN (custom) and ResNet-18 builder.
"""

from loguru import logger
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    """
    Custom CNN with 3 conv blocks, BatchNorm, ReLU, MaxPool, and GAP head.

    Architecture:
        Block 1: Conv(3->32, k=3, p=1) -> BN -> ReLU -> MaxPool(2)
                 Output: (N, 32, 16, 16)
        Block 2: Conv(32->64, k=3, p=1) -> BN -> ReLU -> MaxPool(2)
                 Output: (N, 64, 8, 8)
        Block 3: Conv(64->128, k=3, p=1) -> BN -> ReLU -> MaxPool(2)
                 Output: (N, 128, 4, 4)
        Head: AdaptiveAvgPool(1) -> Flatten -> Linear(128, num_classes)

    Spatial size at GAP input: 4x4 (see Part C1).
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        logger.debug(f"Creating SmallCNN with {num_classes} classes")

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.debug(f"SmallCNN total parameters: {total_params:,}")

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_resnet18(num_classes: int, mode: str) -> nn.Module:
    """
    Build pretrained ResNet-18 with a new head for transfer learning.

    Args:
        num_classes: Number of output classes (5 for CIFAR-10 subset)
        mode: "feature_extract" (freeze backbone) or "finetune" (train all)

    Returns:
        nn.Module with pretrained ResNet-18 + new fc layer.

    Parameter counts:
        - feature_extract: ~2,565 trainable (only the new head)
        - finetune: ~11.2M trainable (entire network)
    """
    logger.debug(f"Building ResNet-18 with {num_classes} classes, mode={mode}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    logger.debug("Loaded pretrained ResNet-18 weights")

    if mode == "feature_extract":
        logger.debug("Freezing backbone parameters (feature extraction mode)")
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    logger.debug(f"Replaced fc layer with Linear(512, {num_classes})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.debug(f"ResNet-18: {trainable:,} trainable params of {total:,} total")

    return model
