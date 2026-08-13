"""
Tests for model definitions: SmallCNN and ResNet-18 builder.
"""

from typing import cast

import torch
from torch import nn

from src.config import NUM_CLASSES, SMALLCNN_CHANNELS
from src.models import SmallCNN, build_resnet18


class TestSmallCNN:
    """Tests for the SmallCNN model."""

    def test_forward_pass(self):
        """SmallCNN should accept a batch and produce logits."""
        model = SmallCNN(num_classes=NUM_CLASSES)
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        logits = model(x)
        assert logits.shape == (batch_size, NUM_CLASSES)

    def test_output_shape(self):
        """SmallCNN should output correct shape for different batch sizes."""
        model = SmallCNN(num_classes=NUM_CLASSES)

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 32, 32)
            logits = model(x)
            assert logits.shape == (batch_size, NUM_CLASSES)

    def test_gap_spatial_size(self):
        """Global Average Pooling should reduce spatial dimensions to 1x1."""
        model = SmallCNN(num_classes=NUM_CLASSES)

        x = torch.randn(1, 3, 32, 32)
        x = model.block1(x)
        x = model.block2(x)
        x = model.block3(x)

        x = model.gap(x)
        assert x.shape == (1, SMALLCNN_CHANNELS[-1], 1, 1)

        x = x.view(x.size(0), -1)
        assert x.shape == (1, SMALLCNN_CHANNELS[-1])

    def test_parameter_count(self):
        """SmallCNN should have approximately 94,341 parameters."""
        model = SmallCNN(num_classes=NUM_CLASSES)
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params == 94341, f"Expected 94,341 params, got {total_params}"

    def test_block_output_shapes(self):
        """Each block should produce the expected output shape."""
        model = SmallCNN(num_classes=NUM_CLASSES)
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        x1 = model.block1(x)
        assert x1.shape == (batch_size, 32, 16, 16)

        x2 = model.block2(x1)
        assert x2.shape == (batch_size, 64, 8, 8)

        x3 = model.block3(x2)
        assert x3.shape == (batch_size, 128, 4, 4)

    def test_batch_norm_training_mode(self):
        """BatchNorm should behave differently in train vs eval mode."""
        model = SmallCNN(num_classes=NUM_CLASSES)
        model.train()
        x = torch.randn(4, 3, 32, 32)

        out_train = model.block1(x)

        model.eval()
        out_eval = model.block1(x)

        assert not torch.allclose(out_train, out_eval, atol=1e-3)


class TestResNet18:
    """Tests for ResNet-18 builder."""

    def test_feature_extract_mode_parameters(self):
        """Feature extraction mode should have ~2,565 trainable params."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="feature_extract")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        assert trainable == 2565, f"Expected 2,565, got {trainable}"
        assert total == 11179077, f"Expected 11,179,077, got {total}"

    def test_finetune_mode_parameters(self):
        """Fine-tuning mode should have all ~11.2M parameters trainable."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        assert trainable == total
        assert total == 11179077, f"Expected 11,179,077, got {total}"

    def test_forward_pass_feature_extract(self):
        """ResNet-18 should work in feature extraction mode."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="feature_extract")
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        logits = model(x)
        assert logits.shape == (batch_size, NUM_CLASSES)

    def test_forward_pass_finetune(self):
        """ResNet-18 should work in fine-tuning mode."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        logits = model(x)
        assert logits.shape == (batch_size, NUM_CLASSES)

    def test_mode_parameter_difference(self):
        """Feature extraction and fine-tuning should have different trainable param counts."""
        model_fe = build_resnet18(num_classes=NUM_CLASSES, mode="feature_extract")
        model_ft = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")

        params_fe = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
        params_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)

        assert params_ft > params_fe * 1000

    def test_fc_layer_replaced(self):
        """The final fc layer should be replaced with correct output size."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")

        # 🔧 FIX: Cast to Linear to help type checker
        fc = cast(nn.Linear, model.fc)
        assert fc.in_features == 512
        assert fc.out_features == NUM_CLASSES

    def test_backbone_frozen_in_feature_extract(self):
        """In feature extraction mode, backbone parameters should be frozen."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="feature_extract")

        # 🔧 FIX: Cast to specific layer types
        conv1 = cast(nn.Conv2d, model.conv1)
        assert conv1.weight.requires_grad is False

        bn1 = cast(nn.BatchNorm2d, model.bn1)
        assert bn1.weight.requires_grad is False

        layer1 = cast(nn.Sequential, model.layer1)
        for param in layer1.parameters():
            assert param.requires_grad is False

        fc = cast(nn.Linear, model.fc)
        assert fc.weight.requires_grad is True
        assert fc.bias.requires_grad is True

    def test_all_parameters_trainable_in_finetune(self):
        """In fine-tuning mode, all parameters should be trainable."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")

        for param in model.parameters():
            assert param.requires_grad is True

    def test_correct_weights_loaded(self):
        """The model should load pretrained weights (not random)."""
        model = build_resnet18(num_classes=NUM_CLASSES, mode="finetune")

        # 🔧 FIX: Cast to specific layer types
        conv1 = cast(nn.Conv2d, model.conv1)
        assert not torch.allclose(conv1.weight, torch.zeros_like(conv1.weight))

        fc = cast(nn.Linear, model.fc)
        assert torch.abs(fc.weight).mean() > 0
