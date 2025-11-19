"""
Neural Style Transfer Suite
A comprehensive neural style transfer implementation with real-time video processing,
multiple model architectures, GPU acceleration, and advanced features.
"""

__version__ = "1.0.0"
__author__ = "AI Medical Diagnosis Team"

from .models.vgg19_model import VGG19StyleTransfer
from .models.resnet_model import ResNetStyleTransfer
from .core.style_transfer import StyleTransfer
from .core.video_processor import VideoStyleTransfer
from .core.webcam_processor import WebcamStyleTransfer
from .core.batch_processor import BatchProcessor
from .utils.gpu_utils import GPUManager

__all__ = [
    'VGG19StyleTransfer',
    'ResNetStyleTransfer',
    'StyleTransfer',
    'VideoStyleTransfer',
    'WebcamStyleTransfer',
    'BatchProcessor',
    'GPUManager',
]
