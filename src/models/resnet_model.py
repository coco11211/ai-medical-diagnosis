"""ResNet-based model for medical imaging analysis."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
from .base_model import BaseModel
from loguru import logger


class MedicalResNet(BaseModel):
    """ResNet model adapted for medical imaging diagnosis."""

    def __init__(
        self,
        num_classes: int,
        resnet_variant: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        device: str = 'cuda',
        freeze_backbone: bool = False
    ):
        """
        Initialize MedicalResNet.

        Args:
            num_classes: Number of output classes
            resnet_variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            device: Device to run model on
            freeze_backbone: Whether to freeze backbone initially
        """
        super().__init__(num_classes, dropout_rate, device)

        # Load pretrained ResNet
        if resnet_variant == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_variant == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_variant == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif resnet_variant == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif resnet_variant == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {resnet_variant}")

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        self.to(self.device)
        logger.info(f"Initialized MedicalResNet ({resnet_variant}) with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings before classification layer.

        Args:
            x: Input tensor

        Returns:
            Feature embeddings
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return features

    def fine_tune(self, num_layers_to_unfreeze: int = 2) -> None:
        """
        Fine-tune by unfreezing last n layers of backbone.

        Args:
            num_layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        # First freeze all
        self.freeze_backbone()

        # Get backbone layers
        backbone_layers = list(self.backbone.children())

        # Unfreeze last n layers
        for layer in backbone_layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        logger.info(f"Fine-tuning: unfroze last {num_layers_to_unfreeze} layers")