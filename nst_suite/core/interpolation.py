"""Style interpolation functionality."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image

from ..core.transfer import FastStyleTransfer
from ..utils.image import load_image, save_image


class StyleInterpolator:
    """
    Style interpolation for smooth transitions between multiple styles.
    """

    def __init__(
        self,
        model_paths: List[str],
        device: Optional[torch.device] = None
    ):
        """
        Initialize style interpolator.

        Args:
            model_paths: List of paths to style transfer models
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load all models
        self.models = []
        for model_path in model_paths:
            model = FastStyleTransfer(model_path, device=self.device)
            model.network.eval()
            self.models.append(model)

        self.num_models = len(self.models)

    def interpolate(
        self,
        content_image: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate between multiple styles.

        Args:
            content_image: Content image as numpy array
            weights: Array of weights for each style (should sum to 1)

        Returns:
            Interpolated stylized image
        """
        if len(weights) != self.num_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({self.num_models})"
            )

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Apply each style
        stylized_images = []
        for model in self.models:
            stylized = model.transfer(content_image)
            stylized_images.append(stylized)

        # Weighted blend
        result = np.zeros_like(content_image, dtype=np.float32)
        for stylized, weight in zip(stylized_images, weights):
            result += stylized.astype(np.float32) * weight

        return result.clip(0, 255).astype(np.uint8)

    def create_transition(
        self,
        content_image: np.ndarray,
        style_a_idx: int,
        style_b_idx: int,
        num_steps: int = 10,
        interpolation: str = 'linear'
    ) -> List[np.ndarray]:
        """
        Create smooth transition between two styles.

        Args:
            content_image: Content image
            style_a_idx: Index of first style
            style_b_idx: Index of second style
            num_steps: Number of interpolation steps
            interpolation: Type of interpolation ('linear', 'ease_in', 'ease_out', 'ease_in_out')

        Returns:
            List of interpolated images
        """
        # Generate interpolation weights
        if interpolation == 'linear':
            alphas = np.linspace(0, 1, num_steps)
        elif interpolation == 'ease_in':
            alphas = np.linspace(0, 1, num_steps) ** 2
        elif interpolation == 'ease_out':
            alphas = 1 - (1 - np.linspace(0, 1, num_steps)) ** 2
        elif interpolation == 'ease_in_out':
            alphas = self._ease_in_out(num_steps)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")

        # Create interpolated images
        interpolated_images = []

        for alpha in alphas:
            weights = np.zeros(self.num_models)
            weights[style_a_idx] = 1 - alpha
            weights[style_b_idx] = alpha

            stylized = self.interpolate(content_image, weights)
            interpolated_images.append(stylized)

        return interpolated_images

    def _ease_in_out(self, num_steps: int) -> np.ndarray:
        """Generate ease-in-out interpolation curve."""
        t = np.linspace(0, 1, num_steps)
        return t * t * (3 - 2 * t)

    def create_morphing_video(
        self,
        content_image: np.ndarray,
        output_path: str,
        style_sequence: Optional[List[int]] = None,
        steps_per_transition: int = 30,
        fps: float = 30.0
    ):
        """
        Create video showing morphing between multiple styles.

        Args:
            content_image: Content image
            output_path: Path to save output video
            style_sequence: Sequence of style indices to transition through
            steps_per_transition: Number of frames for each transition
            fps: Frames per second
        """
        from ..utils.video import VideoProcessor

        if style_sequence is None:
            # Default: cycle through all styles
            style_sequence = list(range(self.num_models)) + [0]

        # Generate all transition frames
        all_frames = []

        for i in range(len(style_sequence) - 1):
            style_a = style_sequence[i]
            style_b = style_sequence[i + 1]

            transition_frames = self.create_transition(
                content_image,
                style_a,
                style_b,
                num_steps=steps_per_transition,
                interpolation='ease_in_out'
            )

            all_frames.extend(transition_frames)

        # Write video
        video_processor = VideoProcessor()
        video_processor.write_video(all_frames, output_path, fps)

        print(f"Morphing video saved to {output_path}")


class FeatureInterpolator:
    """
    Feature-level interpolation for more sophisticated style blending.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        """
        Initialize feature interpolator.

        Args:
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        from ..models.vgg import VGG19StyleTransfer
        self.feature_extractor = VGG19StyleTransfer(device=self.device)

    def interpolate_features(
        self,
        style_features_a: dict,
        style_features_b: dict,
        alpha: float
    ) -> dict:
        """
        Interpolate between two sets of style features.

        Args:
            style_features_a: Features from first style
            style_features_b: Features from second style
            alpha: Interpolation weight (0 = all A, 1 = all B)

        Returns:
            Interpolated features
        """
        interpolated = {}

        for layer in style_features_a.keys():
            feat_a = style_features_a[layer]
            feat_b = style_features_b[layer]

            # Interpolate feature maps
            interpolated[layer] = (1 - alpha) * feat_a + alpha * feat_b

        return interpolated


class SpatialStyleInterpolator:
    """
    Spatially-varying style interpolation for creative effects.
    """

    def __init__(
        self,
        model_paths: List[str],
        device: Optional[torch.device] = None
    ):
        """
        Initialize spatial style interpolator.

        Args:
            model_paths: List of paths to style models
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.models = []
        for model_path in model_paths:
            model = FastStyleTransfer(model_path, device=self.device)
            self.models.append(model)

    def interpolate_spatial(
        self,
        content_image: np.ndarray,
        weight_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply spatially-varying style interpolation.

        Args:
            content_image: Content image (H, W, C)
            weight_map: Weight map (H, W, num_styles) indicating style weights per pixel

        Returns:
            Stylized image with spatially-varying styles
        """
        # Apply each style
        stylized_images = []
        for model in self.models:
            stylized = model.transfer(content_image)
            stylized_images.append(stylized.astype(np.float32))

        # Blend based on weight map
        result = np.zeros_like(content_image, dtype=np.float32)

        for i, stylized in enumerate(stylized_images):
            weights = weight_map[:, :, i:i+1]
            result += stylized * weights

        return result.clip(0, 255).astype(np.uint8)

    def create_gradient_blend(
        self,
        content_image: np.ndarray,
        style_a_idx: int,
        style_b_idx: int,
        direction: str = 'horizontal'
    ) -> np.ndarray:
        """
        Create gradient blend between two styles.

        Args:
            content_image: Content image
            style_a_idx: Index of first style
            style_b_idx: Index of second style
            direction: Gradient direction ('horizontal', 'vertical', 'radial')

        Returns:
            Stylized image with gradient blend
        """
        h, w = content_image.shape[:2]
        num_styles = len(self.models)

        # Create weight map
        weight_map = np.zeros((h, w, num_styles), dtype=np.float32)

        if direction == 'horizontal':
            gradient = np.linspace(0, 1, w)
            weight_map[:, :, style_a_idx] = 1 - gradient
            weight_map[:, :, style_b_idx] = gradient

        elif direction == 'vertical':
            gradient = np.linspace(0, 1, h)[:, np.newaxis]
            weight_map[:, :, style_a_idx] = 1 - gradient
            weight_map[:, :, style_b_idx] = gradient

        elif direction == 'radial':
            y, x = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            gradient = (distance / max_dist).clip(0, 1)
            weight_map[:, :, style_a_idx] = 1 - gradient
            weight_map[:, :, style_b_idx] = gradient

        return self.interpolate_spatial(content_image, weight_map)
