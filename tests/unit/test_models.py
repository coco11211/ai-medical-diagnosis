"""Unit tests for deep learning models."""

import pytest
import torch
from src.models.base_model import BaseModel
from src.models.resnet_model import MedicalResNet
from src.models.efficientnet_model import MedicalEfficientNet


class TestMedicalResNet:
    """Test MedicalResNet class."""

    def test_initialization(self, disease_names, device):
        """Test model initialization."""
        model = MedicalResNet(
            num_classes=len(disease_names),
            resnet_variant='resnet18',  # Use smaller variant for testing
            pretrained=False,
            device=device
        )

        assert model.num_classes == len(disease_names)
        assert model.device.type in ['cuda', 'cpu']

    def test_forward_pass(self, disease_names, sample_tensor, device):
        """Test forward pass."""
        model = MedicalResNet(
            num_classes=len(disease_names),
            resnet_variant='resnet18',
            pretrained=False,
            device=device
        )

        output = model(sample_tensor.to(device))

        assert output.shape == (1, len(disease_names))

    def test_predict(self, disease_names, sample_tensor, device):
        """Test prediction method."""
        model = MedicalResNet(
            num_classes=len(disease_names),
            resnet_variant='resnet18',
            pretrained=False,
            device=device
        )

        predictions, confidences = model.predict(sample_tensor.to(device))

        assert predictions.shape == (1,)
        assert confidences.shape == (1,)

    def test_get_embedding(self, disease_names, sample_tensor, device):
        """Test embedding extraction."""
        model = MedicalResNet(
            num_classes=len(disease_names),
            resnet_variant='resnet18',
            pretrained=False,
            device=device
        )

        embeddings = model.get_embedding(sample_tensor.to(device))

        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1


class TestMedicalEfficientNet:
    """Test MedicalEfficientNet class."""

    def test_initialization(self, disease_names, device):
        """Test model initialization."""
        model = MedicalEfficientNet(
            num_classes=len(disease_names),
            efficientnet_variant='efficientnet-b0',  # Use smaller variant
            pretrained=False,
            device=device
        )

        assert model.num_classes == len(disease_names)

    def test_forward_pass(self, disease_names, sample_tensor, device):
        """Test forward pass."""
        model = MedicalEfficientNet(
            num_classes=len(disease_names),
            efficientnet_variant='efficientnet-b0',
            pretrained=False,
            device=device
        )

        output = model(sample_tensor.to(device))

        assert output.shape == (1, len(disease_names))
