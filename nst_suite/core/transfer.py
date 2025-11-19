"""Main style transfer implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..models.base import BaseStyleModel
from ..models.vgg import VGG19StyleTransfer, FastVGG19StyleTransfer
from ..utils.image import preprocess_image, postprocess_image


class StyleTransfer:
    """
    Main style transfer class that orchestrates the transfer process.
    """

    def __init__(
        self,
        model: Optional[BaseStyleModel] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize style transfer.

        Args:
            model: Pre-trained style model. If None, uses VGG19.
            device: PyTorch device (cuda or cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is None:
            self.model = VGG19StyleTransfer(device=self.device)
        else:
            self.model = model.to_device(self.device)

    def transfer(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Apply style transfer to content image.

        Args:
            content_image: Content image as numpy array
            style_image: Style image as numpy array
            **kwargs: Additional parameters for the transfer method

        Returns:
            Stylized image as numpy array
        """
        raise NotImplementedError("Subclasses must implement transfer method")


class OptimizationBasedTransfer(StyleTransfer):
    """
    Optimization-based style transfer (Gatys et al. 2016).

    This method optimizes the output image directly to match the content
    and style of the input images.
    """

    def __init__(
        self,
        model: Optional[BaseStyleModel] = None,
        device: Optional[torch.device] = None,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        tv_weight: float = 1e-3
    ):
        """
        Initialize optimization-based transfer.

        Args:
            model: Pre-trained style model
            device: PyTorch device
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss
        """
        super().__init__(model, device)

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    def transfer(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        num_steps: int = 300,
        learning_rate: float = 0.03,
        init_method: str = 'content',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Apply optimization-based style transfer.

        Args:
            content_image: Content image as numpy array (H, W, C) in [0, 255]
            style_image: Style image as numpy array (H, W, C) in [0, 255]
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimizer
            init_method: Initialization method ('content', 'style', or 'random')
            show_progress: Whether to show progress bar

        Returns:
            Stylized image as numpy array (H, W, C) in [0, 255]
        """
        # Preprocess images
        content_tensor = preprocess_image(content_image, self.device)
        style_tensor = preprocess_image(style_image, self.device)

        # Initialize output image
        if init_method == 'content':
            output_tensor = content_tensor.clone()
        elif init_method == 'style':
            output_tensor = style_tensor.clone()
        else:
            output_tensor = torch.randn_like(content_tensor)

        output_tensor.requires_grad_(True)

        # Extract target features
        with torch.no_grad():
            content_features = self.model.extract_features(content_tensor)
            style_features = self.model.extract_features(style_tensor)

        # Optimizer
        optimizer = optim.LBFGS([output_tensor], lr=learning_rate, max_iter=20)

        # Optimization loop
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for step in progress_bar:
            def closure():
                optimizer.zero_grad()

                # Extract features from current output
                output_features = self.model.extract_features(output_tensor)

                # Compute losses
                content_loss = self.model.compute_content_loss(
                    content_features, output_features
                )
                style_loss = self.model.compute_style_loss(
                    style_features, output_features
                )
                tv_loss = self.model.compute_total_variation_loss(output_tensor)

                # Total loss
                total_loss = (
                    self.content_weight * content_loss +
                    self.style_weight * style_loss +
                    self.tv_weight * tv_loss
                )

                total_loss.backward()

                # Update progress bar
                progress_bar.set_postfix({
                    'content': f'{content_loss.item():.2f}',
                    'style': f'{style_loss.item():.2f}',
                    'tv': f'{tv_loss.item():.4f}'
                })

                return total_loss

            optimizer.step(closure)

            # Clamp values to valid range
            with torch.no_grad():
                output_tensor.clamp_(0, 1)

        # Postprocess and return
        return postprocess_image(output_tensor)


class FastStyleTransfer(StyleTransfer):
    """
    Fast feed-forward style transfer.

    This method uses a pre-trained feed-forward network to perform
    style transfer in a single forward pass.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize fast style transfer.

        Args:
            model_path: Path to pre-trained fast transfer model
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize fast transfer network
        self.network = FastVGG19StyleTransfer(device=self.device)

        # Load pre-trained weights if provided
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load pre-trained model weights.

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.network.load_state_dict(checkpoint)

        self.network.eval()

    def save_model(self, model_path: str):
        """
        Save model weights.

        Args:
            model_path: Path to save model checkpoint
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, model_path)

    def transfer(
        self,
        content_image: np.ndarray,
        style_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply fast style transfer.

        Args:
            content_image: Content image as numpy array (H, W, C) in [0, 255]
            style_image: Not used for fast transfer (model is style-specific)
            **kwargs: Additional parameters (ignored)

        Returns:
            Stylized image as numpy array (H, W, C) in [0, 255]
        """
        # Preprocess image
        content_tensor = preprocess_image(content_image, self.device)

        # Apply style transfer
        with torch.no_grad():
            output_tensor = self.network(content_tensor)

        # Postprocess and return
        return postprocess_image(output_tensor)

    def transfer_batch(
        self,
        content_images: list
    ) -> list:
        """
        Apply style transfer to a batch of images.

        Args:
            content_images: List of content images as numpy arrays

        Returns:
            List of stylized images as numpy arrays
        """
        # Preprocess all images
        content_tensors = [
            preprocess_image(img, self.device) for img in content_images
        ]
        content_batch = torch.cat(content_tensors, dim=0)

        # Apply style transfer
        with torch.no_grad():
            output_batch = self.network(content_batch)

        # Postprocess and return
        stylized_images = []
        for i in range(output_batch.size(0)):
            output_tensor = output_batch[i:i+1]
            stylized_image = postprocess_image(output_tensor)
            stylized_images.append(stylized_image)

        return stylized_images


class MultiStyleFastTransfer(FastStyleTransfer):
    """
    Fast style transfer with multiple styles and interpolation.
    """

    def __init__(
        self,
        num_styles: int = 1,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-style fast transfer.

        Args:
            num_styles: Number of styles the model supports
            model_path: Path to pre-trained model
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        from ..models.resnet import AdaptiveResNetStyleTransfer
        self.network = AdaptiveResNetStyleTransfer(
            num_styles=num_styles,
            device=self.device
        )
        self.num_styles = num_styles

        if model_path is not None:
            self.load_model(model_path)

    def transfer(
        self,
        content_image: np.ndarray,
        style_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply multi-style transfer with interpolation.

        Args:
            content_image: Content image as numpy array (H, W, C) in [0, 255]
            style_weights: Array of style weights (num_styles,). If None, uses equal weights.
            **kwargs: Additional parameters (ignored)

        Returns:
            Stylized image as numpy array (H, W, C) in [0, 255]
        """
        # Preprocess image
        content_tensor = preprocess_image(content_image, self.device)

        # Prepare style weights
        if style_weights is not None:
            style_weights_tensor = torch.tensor(
                style_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            style_weights_tensor = None

        # Apply style transfer
        with torch.no_grad():
            output_tensor = self.network(content_tensor, style_weights_tensor)

        # Postprocess and return
        return postprocess_image(output_tensor)
