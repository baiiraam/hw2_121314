"""
Custom implementations of Mixup and CutMix augmentation.
No external libraries used - self-implemented per assignment rules.
"""

import numpy as np
import torch
from loguru import logger


def mixup_batch(x, y, alpha: float = 1.0):
    """
    Mixup: convex combination of two images and labels.

    Args:
        x: (N, C, H, W) batch of images
        y: (N,) batch of labels
        alpha: Beta distribution parameter

    Returns:
        x_mixed: (N, C, H, W) mixed images
        y_a: (N,) first set of labels
        y_b: (N,) second set of labels
        lam: scalar mixing coefficient
    """
    logger.debug(f"Applying Mixup with alpha={alpha}")

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size)

    x_mixed = lam * x + (1 - lam) * x[idx]

    logger.debug(f"Mixup: lam={lam:.4f}, batch_size={batch_size}")
    return x_mixed, y, y[idx], lam


def cutmix_batch(x, y, alpha: float = 1.0):
    """
    CutMix: paste random box from one image onto another.

    Args:
        x: (N, C, H, W) batch of images
        y: (N,) batch of labels
        alpha: Beta distribution parameter

    Returns:
        x_mixed: (N, C, H, W) mixed images
        y_a: (N,) original labels
        y_b: (N,) patch labels
        lam: mixing coefficient (based on box area)

    Note: Uses clipped box dimensions to avoid np.random.randint edge cases.
    """
    batch_size, _, H, W = x.size()
    logger.debug(f"Applying CutMix with alpha={alpha}, H={H}, W={W}")

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)

    # Calculate box dimensions with guard against edge cases
    box_w = int(W * np.sqrt(1 - lam))
    box_w = min(W - 1, max(1, box_w))

    box_h = int(H * np.sqrt(1 - lam))
    box_h = min(H - 1, max(1, box_h))

    # Random box position (top-left corner)
    cx = np.random.randint(0, W - box_w + 1)
    cy = np.random.randint(0, H - box_h + 1)

    # Create mixed image
    x_mixed = x.clone()
    x_mixed[:, :, cy : cy + box_h, cx : cx + box_w] = x[
        idx, :, cy : cy + box_h, cx : cx + box_w
    ]

    lam_actual = 1 - (box_w * box_h) / (W * H)

    logger.debug(
        f"CutMix: box=({cx},{cy},{cx + box_w},{cy + box_h}), lam={lam_actual:.4f}"
    )
    return x_mixed, y, y[idx], lam_actual


def mix_criterion(loss_fn, logits, y_a, y_b, lam):
    """
    Combined loss for Mixup/CutMix.

    Why convex combination (Part C3):
    The input is a mix of two images, so the correct output is a mix of
    two labels. A single cross-entropy would only work for one label.
    The convex combination properly handles the mixed supervision.
    """
    logger.debug(f"Computing mixed loss: lam={lam:.4f}")
    return lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
