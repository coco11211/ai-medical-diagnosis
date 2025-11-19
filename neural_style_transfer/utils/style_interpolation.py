"""
Style Interpolation Utilities
Enables blending multiple styles and smooth transitions
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from PIL import Image
import numpy as np


class StyleInterpolator:
    """
    Handles style interpolation and blending between multiple styles
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize style interpolator

        Args:
            device: Device to run computations on
        """
        self.device = device

    def interpolate_features(self, features1: torch.Tensor, features2: torch.Tensor,
                            alpha: float) -> torch.Tensor:
        """
        Linearly interpolate between two feature tensors

        Args:
            features1: First feature tensor
            features2: Second feature tensor
            alpha: Interpolation factor (0.0 = features1, 1.0 = features2)

        Returns:
            Interpolated features
        """
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        return (1 - alpha) * features1 + alpha * features2

    def blend_styles(self, style_features: List[Dict[str, torch.Tensor]],
                    weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Blend multiple style features with given weights

        Args:
            style_features: List of style feature dictionaries
            weights: Optional list of weights (normalized automatically)

        Returns:
            Blended style features
        """
        if not style_features:
            raise ValueError("No style features provided")

        if weights is None:
            weights = [1.0 / len(style_features)] * len(style_features)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Blend features for each layer
        blended = {}

        for layer_name in style_features[0].keys():
            layer_features = []

            for features, weight in zip(style_features, weights):
                layer_features.append(features[layer_name] * weight)

            blended[layer_name] = sum(layer_features)

        return blended

    def create_transition_sequence(self, style1_features: Dict[str, torch.Tensor],
                                   style2_features: Dict[str, torch.Tensor],
                                   num_frames: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create smooth transition sequence between two styles

        Args:
            style1_features: Starting style features
            style2_features: Ending style features
            num_frames: Number of intermediate frames

        Returns:
            List of interpolated style features
        """
        sequence = []

        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0.0

            interpolated = {}
            for layer_name in style1_features.keys():
                interpolated[layer_name] = self.interpolate_features(
                    style1_features[layer_name],
                    style2_features[layer_name],
                    alpha
                )

            sequence.append(interpolated)

        return sequence

    def blend_with_easing(self, features1: torch.Tensor, features2: torch.Tensor,
                         alpha: float, easing: str = 'linear') -> torch.Tensor:
        """
        Interpolate with easing function

        Args:
            features1: First feature tensor
            features2: Second feature tensor
            alpha: Interpolation factor [0, 1]
            easing: Easing function ('linear', 'ease_in', 'ease_out', 'ease_in_out')

        Returns:
            Interpolated features
        """
        # Apply easing function
        if easing == 'ease_in':
            alpha = alpha ** 2
        elif easing == 'ease_out':
            alpha = 1 - (1 - alpha) ** 2
        elif easing == 'ease_in_out':
            if alpha < 0.5:
                alpha = 2 * alpha ** 2
            else:
                alpha = 1 - 2 * (1 - alpha) ** 2
        # 'linear' uses alpha as-is

        return self.interpolate_features(features1, features2, alpha)

    def create_style_palette(self, styles: List[Dict[str, torch.Tensor]],
                            mix_matrix: np.ndarray) -> List[Dict[str, torch.Tensor]]:
        """
        Create palette of blended styles using mix matrix

        Args:
            styles: List of base style features
            mix_matrix: Matrix of shape (output_styles, input_styles) with blend weights

        Returns:
            List of blended style palettes
        """
        palette = []

        for weights in mix_matrix:
            blended = self.blend_styles(styles, weights.tolist())
            palette.append(blended)

        return palette


class StyleMixer:
    """
    Advanced style mixing with spatial control
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def spatial_style_transfer(self, content: torch.Tensor,
                              style_features: List[Dict[str, torch.Tensor]],
                              masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply different styles to different regions

        Args:
            content: Content image tensor
            style_features: List of style feature dicts
            masks: List of binary masks for each style

        Returns:
            Combined style features
        """
        if len(style_features) != len(masks):
            raise ValueError("Number of styles must match number of masks")

        combined = {}
        layer_names = style_features[0].keys()

        for layer_name in layer_names:
            layer_combined = torch.zeros_like(style_features[0][layer_name])

            for features, mask in zip(style_features, masks):
                # Resize mask to match feature dimensions
                _, _, h, w = features[layer_name].shape
                resized_mask = nn.functional.interpolate(
                    mask,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )

                # Apply mask
                layer_combined += features[layer_name] * resized_mask

            combined[layer_name] = layer_combined

        return combined

    def create_gradient_blend(self, features1: Dict[str, torch.Tensor],
                             features2: Dict[str, torch.Tensor],
                             direction: str = 'horizontal') -> Dict[str, torch.Tensor]:
        """
        Create gradient blend between two styles

        Args:
            features1: First style features
            features2: Second style features
            direction: 'horizontal' or 'vertical'

        Returns:
            Gradient blended features
        """
        blended = {}

        for layer_name in features1.keys():
            _, _, h, w = features1[layer_name].shape

            # Create gradient mask
            if direction == 'horizontal':
                gradient = torch.linspace(0, 1, w, device=self.device)
                gradient = gradient.view(1, 1, 1, w).expand(1, 1, h, w)
            else:  # vertical
                gradient = torch.linspace(0, 1, h, device=self.device)
                gradient = gradient.view(1, 1, h, 1).expand(1, 1, h, w)

            # Blend using gradient
            blended[layer_name] = (
                features1[layer_name] * (1 - gradient) +
                features2[layer_name] * gradient
            )

        return blended


class TemporalStyleSmoother:
    """
    Smooth style transitions over time for video processing
    """

    def __init__(self, temporal_weight: float = 0.5):
        """
        Initialize temporal smoother

        Args:
            temporal_weight: Weight for previous frame (0-1)
        """
        self.temporal_weight = temporal_weight
        self.previous_output = None

    def smooth(self, current_output: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing

        Args:
            current_output: Current frame output

        Returns:
            Smoothed output
        """
        if self.previous_output is None:
            self.previous_output = current_output.clone()
            return current_output

        # Blend with previous frame
        smoothed = (
            self.temporal_weight * self.previous_output +
            (1 - self.temporal_weight) * current_output
        )

        self.previous_output = smoothed.clone()
        return smoothed

    def reset(self):
        """Reset smoother state"""
        self.previous_output = None
