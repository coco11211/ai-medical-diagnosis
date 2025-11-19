"""
Neural Style Transfer Suite
============================

A comprehensive neural style transfer suite for Windows 11 with:
- Real-time video style transfer
- Multiple style models (VGG19, ResNet)
- Custom style training pipeline
- Batch processing
- GPU acceleration (CUDA)
- Style interpolation
- Webcam integration
- 4K video support
- Style library management
- Export to multiple formats

Author: Neural Style Transfer Suite Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Neural Style Transfer Suite Team"

from .core.transfer import StyleTransfer
from .models.vgg import VGG19StyleTransfer
from .models.resnet import ResNetStyleTransfer

__all__ = [
    "StyleTransfer",
    "VGG19StyleTransfer",
    "ResNetStyleTransfer",
]
