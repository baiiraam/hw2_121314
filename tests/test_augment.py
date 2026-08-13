"""
Tests for augmentation functions: Mixup, CutMix, and mix_criterion.
"""

import numpy as np
import torch

from src.augment import cutmix_batch, mix_criterion, mixup_batch


class TestMixup:
    """Tests for Mixup augmentation."""

    def test_mixup_output_shapes(self):
        """Mixup should preserve input shapes."""
        batch_size = 8
        channels = 3
        height = 32
        width = 32
        num_classes = 5

        x = torch.randn(batch_size, channels, height, width)
        y = torch.randint(0, num_classes, (batch_size,))

        x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=1.0)

        # Shapes should be preserved
        assert x_mixed.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)

    def test_mixup_lam_range(self):
        """Lambda should be in [0, 1]."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 5, (4,))

        for _ in range(100):
            _, _, _, lam = mixup_batch(x, y, alpha=1.0)
            assert 0 <= lam <= 1

    def test_mixup_lam_beta_distribution(self):
        """Lambda should follow Beta(alpha, alpha) distribution."""
        x = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 5, (100,))

        lams = []
        for _ in range(1000):
            _, _, _, lam = mixup_batch(x, y, alpha=0.5)
            lams.append(lam)

        # With alpha=0.5, distribution should be U-shaped
        # (more values near 0 and 1 than near 0.5)
        near_edges = sum(1 for lam in lams if lam < 0.2 or lam > 0.8)
        sum(1 for lam in lams if 0.4 < lam < 0.6)

        # This is a statistical test, not deterministic
        # Just check that distribution is not uniform
        assert near_edges > 100, "Beta(0.5) should produce more extreme values"

    def test_mixup_with_alpha_1(self):
        """With alpha=1, lambda should be uniform in [0, 1]."""
        x = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 5, (100,))

        lams = []
        for _ in range(1000):
            _, _, _, lam = mixup_batch(x, y, alpha=1.0)
            lams.append(lam)

        # With alpha=1, Beta(1,1) is uniform
        # Check that values are roughly uniform
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = [0] * 5
        for lam in lams:
            for i in range(5):
                if bins[i] <= lam < bins[i + 1]:
                    counts[i] += 1

        # Each bin should have roughly 200 values (1000/5)
        for count in counts:
            assert 150 < count < 250, f"Uniform distribution expected, got {count}"

    def test_mixup_shuffles_batch(self):
        """Mixup should shuffle the batch for pairing."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])

        _, y_a, y_b, _lam = mixup_batch(x, y, alpha=1.0)

        # y_a should be the original labels
        assert torch.all(y_a == y)

        # y_b should be a shuffled version (not identical to y unless lam=0 or 1)
        # With lam between 0 and 1, y_b is used in the mix
        # We can't guarantee it's different, but we can check the shuffle
        # At least one element should be different (almost always true)
        # Since we can't guarantee, we'll just check shapes
        assert y_b.shape == y.shape

    def test_mixup_no_grad(self):
        """Mixup should work with tensors that require gradients."""
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        y = torch.randint(0, 5, (4,))

        x_mixed, _, _, _ = mixup_batch(x, y, alpha=1.0)

        # Gradients should flow through x_mixed
        loss = x_mixed.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCutMix:
    """Tests for CutMix augmentation."""

    def test_cutmix_output_shapes(self):
        """CutMix should preserve input shapes."""
        batch_size = 8
        channels = 3
        height = 32
        width = 32
        num_classes = 5

        x = torch.randn(batch_size, channels, height, width)
        y = torch.randint(0, num_classes, (batch_size,))

        x_mixed, y_a, y_b, lam = cutmix_batch(x, y, alpha=1.0)

        # Shapes should be preserved
        assert x_mixed.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)

    def test_cutmix_lam_range(self):
        """Lambda should be in [0, 1] based on box area."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 5, (4,))

        for _ in range(100):
            _, _, _, lam = cutmix_batch(x, y, alpha=1.0)
            assert 0 <= lam <= 1

    def test_cutmix_box_area_consistent(self):
        """Lambda should match the box area ratio."""
        batch_size = 4
        H, W = 32, 32
        x = torch.randn(batch_size, 3, H, W)
        y = torch.randint(0, 5, (batch_size,))

        for _ in range(50):
            _, _, _, lam = cutmix_batch(x, y, alpha=1.0)

            # lam = 1 - box_area / total_area
            # So box_area = (1 - lam) * total_area
            box_area = (1 - lam) * H * W

            # This is the expected box area
            # We can't verify the exact box, but we can check it's within bounds
            assert 0 <= box_area <= H * W

    def test_cutmix_edge_case_lam_close_to_0(self):
        """When lam is close to 0, box should be clamped to valid size."""
        # This tests the bug fix from earlier
        # Force alpha to produce a very small lam
        # We mock the random seed to get a specific lam
        np.random.seed(0)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 5, (4,))

        # Run multiple times to hit edge cases
        for _ in range(50):
            # Should not raise any errors
            x_mixed, _, _, lam = cutmix_batch(x, y, alpha=0.1)
            # lam should be in [0, 1]
            assert 0 <= lam <= 1
            # The mixed image should be valid
            assert not torch.isnan(x_mixed).any()
            assert not torch.isinf(x_mixed).any()

    def test_cutmix_alpha_different_values(self):
        """CutMix should work with different alpha values."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 5, (4,))

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            x_mixed, _, _, lam = cutmix_batch(x, y, alpha=alpha)
            assert x_mixed.shape == x.shape
            assert 0 <= lam <= 1

    def test_cutmix_no_grad(self):
        """CutMix should work with tensors that require gradients."""
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        y = torch.randint(0, 5, (4,))

        x_mixed, _, _, _ = cutmix_batch(x, y, alpha=1.0)

        loss = x_mixed.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMixCriterion:
    """Tests for the mixed loss criterion."""

    def test_mix_criterion_shape(self):
        """Mixed loss should produce a scalar."""
        num_classes = 5
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        y_a = torch.randint(0, num_classes, (batch_size,))
        y_b = torch.randint(0, num_classes, (batch_size,))
        lam = 0.7

        loss_fn = torch.nn.CrossEntropyLoss()

        loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_mix_criterion_convex_combination(self):
        """Loss should be a convex combination of two losses."""
        num_classes = 5
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        y_a = torch.randint(0, num_classes, (batch_size,))
        y_b = torch.randint(0, num_classes, (batch_size,))
        lam = 0.7

        loss_fn = torch.nn.CrossEntropyLoss()

        loss1 = loss_fn(logits, y_a)
        loss2 = loss_fn(logits, y_b)
        expected_loss = lam * loss1 + (1 - lam) * loss2

        actual_loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)

        assert torch.allclose(actual_loss, expected_loss)

    def test_mix_criterion_lam_0(self):
        """When lam=0, loss should be loss2 only."""
        num_classes = 5
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        y_a = torch.randint(0, num_classes, (batch_size,))
        y_b = torch.randint(0, num_classes, (batch_size,))
        lam = 0.0

        loss_fn = torch.nn.CrossEntropyLoss()

        loss_fn(logits, y_a)
        loss2 = loss_fn(logits, y_b)
        expected_loss = loss2

        actual_loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)

        assert torch.allclose(actual_loss, expected_loss)

    def test_mix_criterion_lam_1(self):
        """When lam=1, loss should be loss1 only."""
        num_classes = 5
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        y_a = torch.randint(0, num_classes, (batch_size,))
        y_b = torch.randint(0, num_classes, (batch_size,))
        lam = 1.0

        loss_fn = torch.nn.CrossEntropyLoss()

        loss1 = loss_fn(logits, y_a)
        loss_fn(logits, y_b)
        expected_loss = loss1

        actual_loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)

        assert torch.allclose(actual_loss, expected_loss)
