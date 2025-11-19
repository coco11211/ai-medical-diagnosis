"""Multi-modal fusion for combining imaging and text data."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from loguru import logger


class MultiModalFusion(nn.Module):
    """Fuse image and text features for multi-modal medical diagnosis."""

    def __init__(
        self,
        image_feature_dim: int = 512,
        text_feature_dim: int = 768,
        fusion_method: str = 'concat',
        num_classes: int = 10,
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize multi-modal fusion module.

        Args:
            image_feature_dim: Dimension of image features
            text_feature_dim: Dimension of text features
            fusion_method: Fusion method ('concat', 'add', 'multiply', 'attention')
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            device: Device to run on
        """
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Feature projection layers to match dimensions
        if fusion_method in ['add', 'multiply', 'attention']:
            common_dim = max(image_feature_dim, text_feature_dim)
            self.image_projection = nn.Linear(image_feature_dim, common_dim)
            self.text_projection = nn.Linear(text_feature_dim, common_dim)
            fusion_dim = common_dim
        elif fusion_method == 'concat':
            self.image_projection = nn.Identity()
            self.text_projection = nn.Identity()
            fusion_dim = image_feature_dim + text_feature_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Attention mechanism for fusion
        if fusion_method == 'attention':
            self.attention = CrossModalAttention(common_dim)

        # Build fusion classifier
        layers = []
        prev_dim = fusion_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        self.to(self.device)
        logger.info(f"Initialized MultiModalFusion with {fusion_method} fusion method")

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with multi-modal fusion.

        Args:
            image_features: Image feature tensor (batch_size, image_feature_dim)
            text_features: Text feature tensor (batch_size, text_feature_dim)

        Returns:
            Class logits (batch_size, num_classes)
        """
        # Project features
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat([image_proj, text_proj], dim=1)
        elif self.fusion_method == 'add':
            fused = image_proj + text_proj
        elif self.fusion_method == 'multiply':
            fused = image_proj * text_proj
        elif self.fusion_method == 'attention':
            fused = self.attention(image_proj, text_proj)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Classify
        logits = self.classifier(fused)

        return logits

    def predict(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        return_probabilities: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Args:
            image_features: Image features
            text_features: Text features
            return_probabilities: Whether to return probabilities

        Returns:
            Tuple of (predictions, confidences)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_features, text_features)

            if return_probabilities:
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            else:
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(logits, dim=1)[0]

        return predictions, confidences

    def get_fused_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fused features without classification.

        Args:
            image_features: Image features
            text_features: Text features

        Returns:
            Fused feature tensor
        """
        self.eval()
        with torch.no_grad():
            # Project features
            image_proj = self.image_projection(image_features)
            text_proj = self.text_projection(text_features)

            # Fuse features
            if self.fusion_method == 'concat':
                fused = torch.cat([image_proj, text_proj], dim=1)
            elif self.fusion_method == 'add':
                fused = image_proj + text_proj
            elif self.fusion_method == 'multiply':
                fused = image_proj * text_proj
            elif self.fusion_method == 'attention':
                fused = self.attention(image_proj, text_proj)

        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusion."""

    def __init__(self, feature_dim: int, num_heads: int = 8):
        """
        Initialize cross-modal attention.

        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.

        Args:
            image_features: Image feature tensor
            text_features: Text feature tensor

        Returns:
            Attention-fused features
        """
        batch_size = image_features.shape[0]

        # Expand dimensions for attention
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)  # (B, 1, D)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)  # (B, 1, D)

        # Concatenate features
        combined = torch.cat([image_features, text_features], dim=1)  # (B, 2, D)

        # Compute Q, K, V
        Q = self.query(combined)  # (B, 2, D)
        K = self.key(combined)
        V = self.value(combined)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Average over sequence dimension
        output = attended.mean(dim=1)  # (B, D)

        # Output projection and residual connection
        output = self.out_proj(output)
        output = self.layer_norm(output + image_features.squeeze(1) + text_features.squeeze(1))

        return output


class EarlyFusion(nn.Module):
    """Early fusion - combine raw inputs before processing."""

    def __init__(
        self,
        image_model: nn.Module,
        text_model: nn.Module,
        num_classes: int,
        fusion_dim: int = 1024,
        dropout_rate: float = 0.3
    ):
        """
        Initialize early fusion.

        Args:
            image_model: Image feature extractor
            text_model: Text feature extractor
            num_classes: Number of classes
            fusion_dim: Fusion layer dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.image_model = image_model
        self.text_model = text_model

        # Get feature dimensions
        self.image_feature_dim = getattr(image_model, 'feature_dim', 512)
        self.text_feature_dim = getattr(text_model, 'feature_dim', 768)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.text_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with early fusion."""
        # Extract features
        image_features = self.image_model.get_embedding(images)
        text_features = self.text_model(texts)

        # Concatenate
        combined = torch.cat([image_features, text_features], dim=1)

        # Classify
        logits = self.fusion(combined)

        return logits


class LateFusion(nn.Module):
    """Late fusion - combine predictions from separate models."""

    def __init__(
        self,
        image_model: nn.Module,
        text_model: nn.Module,
        fusion_method: str = 'average'
    ):
        """
        Initialize late fusion.

        Args:
            image_model: Image classification model
            text_model: Text classification model
            fusion_method: How to combine predictions ('average', 'weighted', 'max')
        """
        super().__init__()

        self.image_model = image_model
        self.text_model = text_model
        self.fusion_method = fusion_method

        # Learnable weights for weighted fusion
        if fusion_method == 'weighted':
            self.image_weight = nn.Parameter(torch.tensor(0.5))
            self.text_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with late fusion."""
        # Get predictions from each model
        image_logits = self.image_model(images)
        text_logits = self.text_model(texts)

        # Fuse predictions
        if self.fusion_method == 'average':
            logits = (image_logits + text_logits) / 2
        elif self.fusion_method == 'weighted':
            # Normalize weights
            total_weight = self.image_weight + self.text_weight
            logits = (self.image_weight * image_logits + self.text_weight * text_logits) / total_weight
        elif self.fusion_method == 'max':
            logits = torch.max(image_logits, text_logits)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return logits
