"""Base model class for medical diagnosis models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


class BaseModel(nn.Module, ABC):
    """Abstract base class for medical diagnosis models."""

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize base model.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            device: Device to run model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initialized {self.__class__.__name__} with {num_classes} classes on {self.device}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def save_model(self, path: str) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        }, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from file.

        Args:
            path: Path to model file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings before classification layer.

        Args:
            x: Input tensor

        Returns:
            Feature embeddings
        """
        # This should be overridden in child classes
        raise NotImplementedError("get_embedding must be implemented in child class")

    def predict(
        self,
        x: torch.Tensor,
        return_probabilities: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input data.

        Args:
            x: Input tensor
            return_probabilities: Whether to return probabilities (softmax)

        Returns:
            Tuple of (predictions, confidence scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            if return_probabilities:
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            else:
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(logits, dim=1)[0]

        return predictions, confidences

    def get_model_summary(self) -> Dict[str, any]:
        """Get model summary information."""
        return {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_parameters(),
            'dropout_rate': self.dropout_rate,
            'device': str(self.device)
        }
