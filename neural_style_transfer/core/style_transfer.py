"""
Core Neural Style Transfer Engine
Orchestrates the style transfer process using VGG19 or ResNet models
"""

import torch
import torch.optim as optim
from PIL import Image
from typing import Optional, Tuple, Dict
import logging
from tqdm import tqdm

from ..models.vgg19_model import VGG19StyleTransfer
from ..models.resnet_model import ResNetStyleTransfer
from ..utils.gpu_utils import GPUManager
from ..utils.image_utils import ImageProcessor


class StyleTransfer:
    """
    Main style transfer engine supporting optimization-based and feed-forward methods
    """

    def __init__(self, model_type: str = 'vgg19', device: Optional[str] = None):
        """
        Initialize style transfer engine

        Args:
            model_type: 'vgg19' for optimization-based or 'resnet' for fast feed-forward
            device: Device to run on (auto-detected if None)
        """
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_device()
        self.model_type = model_type

        # Initialize model
        if model_type == 'vgg19':
            self.model = VGG19StyleTransfer(device=self.device)
            logging.info("Initialized VGG19 style transfer model")
        elif model_type == 'resnet':
            self.model = ResNetStyleTransfer(device=self.device)
            self.model = self.gpu_manager.optimize_for_inference(self.model)
            logging.info("Initialized ResNet style transfer model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.image_processor = ImageProcessor()

    def transfer_style_optimization(self,
                                   content_image: Image.Image,
                                   style_image: Image.Image,
                                   num_steps: int = 300,
                                   content_weight: float = 1.0,
                                   style_weight: float = 1000000.0,
                                   tv_weight: float = 0.01,
                                   learning_rate: float = 0.003,
                                   size: Optional[int] = None) -> Image.Image:
        """
        Optimization-based style transfer (VGG19)

        Args:
            content_image: Content PIL Image
            style_image: Style PIL Image
            num_steps: Number of optimization steps
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss (smoothness)
            learning_rate: Learning rate for optimizer
            size: Optional max dimension size

        Returns:
            Stylized PIL Image
        """
        if self.model_type != 'vgg19':
            raise ValueError("Optimization-based transfer requires VGG19 model")

        # Resize images if needed
        if size:
            content_image = self.image_processor.resize_image(content_image, size)
            style_image = self.image_processor.resize_image(style_image, size)

        # Preprocess images
        content_tensor = self.model.preprocess_image(content_image)
        style_tensor = self.model.preprocess_image(style_image, content_tensor.shape[2:])

        # Extract features
        with torch.no_grad():
            content_features, _ = self.model.extract_features(content_tensor)
            _, style_features = self.model.extract_features(style_tensor)

            # Compute style Gram matrices
            style_grams = {
                layer: self.model.gram_matrix(features)
                for layer, features in style_features.items()
            }

        # Initialize target image (start from content image)
        target = content_tensor.clone().requires_grad_(True)

        # Optimizer
        optimizer = optim.LBFGS([target], lr=learning_rate, max_iter=20)

        # Optimization loop
        step = [0]
        pbar = tqdm(total=num_steps, desc="Style Transfer")

        def closure():
            optimizer.zero_grad()

            # Clamp values
            target.data.clamp_(0, 1)

            # Extract target features
            target_content, target_style = self.model.extract_features(target)

            # Content loss
            content_loss = self.model.compute_content_loss(content_features, target_content)
            content_loss *= content_weight

            # Style loss
            style_loss = 0.0
            for layer in self.model.STYLE_LAYERS:
                target_gram = self.model.gram_matrix(target_style[layer])
                style_loss += torch.nn.functional.mse_loss(target_gram, style_grams[layer])
            style_loss *= style_weight

            # Total variation loss for smoothness
            tv_loss = (
                torch.sum(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])) +
                torch.sum(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]))
            )
            tv_loss *= tv_weight

            # Total loss
            loss = content_loss + style_loss + tv_loss

            loss.backward()

            step[0] += 1
            if step[0] % 10 == 0:
                pbar.set_postfix({
                    'total': f'{loss.item():.2f}',
                    'content': f'{content_loss.item():.2f}',
                    'style': f'{style_loss.item():.2f}'
                })
                pbar.update(10)

            return loss

        while step[0] < num_steps:
            optimizer.step(closure)

        pbar.close()

        # Convert result to image
        output_image = self.model.postprocess_image(target)

        return output_image

    def transfer_style_fast(self, content_image: Image.Image,
                           size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Fast feed-forward style transfer (ResNet)
        Requires pre-trained model

        Args:
            content_image: Content PIL Image
            size: Optional target size

        Returns:
            Stylized PIL Image
        """
        if self.model_type != 'resnet':
            raise ValueError("Fast transfer requires ResNet model")

        return self.model.stylize_image(content_image, size)

    def transfer(self, content_image: Image.Image, style_image: Optional[Image.Image] = None,
                **kwargs) -> Image.Image:
        """
        Unified interface for style transfer

        Args:
            content_image: Content image
            style_image: Style image (required for VGG19, ignored for ResNet)
            **kwargs: Additional arguments for specific method

        Returns:
            Stylized image
        """
        if self.model_type == 'vgg19':
            if style_image is None:
                raise ValueError("Style image required for VGG19 model")
            return self.transfer_style_optimization(content_image, style_image, **kwargs)
        else:
            return self.transfer_style_fast(content_image, **kwargs)

    def get_gpu_info(self) -> str:
        """Get GPU information"""
        return self.gpu_manager.get_cuda_info()

    def clear_cache(self):
        """Clear GPU cache"""
        self.gpu_manager.clear_cache()


class OptimizedStyleTransfer:
    """
    Highly optimized style transfer for production use
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.gpu_manager = GPUManager()

        # Use fast ResNet model
        self.model = ResNetStyleTransfer(device=device)
        self.model = self.gpu_manager.optimize_for_inference(self.model)

        # Enable mixed precision if available
        self.use_amp = self.gpu_manager.enable_mixed_precision()

    def process(self, image: Image.Image, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Process image with optimizations

        Args:
            image: Input image
            size: Optional target size

        Returns:
            Stylized image
        """
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    return self.model.stylize_image(image, size)
            else:
                return self.model.stylize_image(image, size)
