"""EfficientNet-based model for medical imaging analysis."""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional
from .base_model import BaseModel
from loguru import logger


class MedicalEfficientNet(BaseModel):
    """EfficientNet model adapted for medical imaging diagnosis."""

    def __init__(
        self,
        num_classes: int,
        efficientnet_variant: str = 'efficientnet-b4',
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        device: str = 'cuda',
        freeze_backbone: bool = False
    ):
        """
        Initialize MedicalEfficientNet.

        Args:
            num_classes: Number of output classes
            efficientnet_variant: EfficientNet variant ('efficientnet-b0' to 'efficientnet-b7')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            device: Device to run model on
            freeze_backbone: Whether to freeze backbone initially
        """
        super().__init__(num_classes, dropout_rate, device)

        # Load pretrained EfficientNet
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(efficientnet_variant)
        else:
            self.backbone = EfficientNet.from_name(efficientnet_variant)

        # Get feature dimension
        feature_dim = self.backbone._fc.in_features

        # Remove the final fully connected layer
        self.backbone._fc = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
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
        logger.info(f"Initialized MedicalEfficientNet ({efficientnet_variant}) with {self.get_num_parameters():,} parameters")

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
        return features

    def fine_tune(self, num_layers_to_unfreeze: int = 2) -> None:
        """
        Fine-tune by unfreezing last n blocks of backbone.

        Args:
            num_layers_to_unfreeze: Number of blocks to unfreeze from the end
        """
        # First freeze all
        self.freeze_backbone()

        # Get backbone blocks
        blocks = list(self.backbone._blocks)

        # Unfreeze last n blocks
        for block in blocks[-num_layers_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        logger.info(f"Fine-tuning: unfroze last {num_layers_to_unfreeze} blocks")

    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Set Swish activation to be memory efficient or not.

        Args:
            memory_efficient: Whether to use memory efficient Swish
        """
        self.backbone.set_swish(memory_efficient=memory_efficient)
        logger.info(f"Set Swish memory_efficient={memory_efficient}")
