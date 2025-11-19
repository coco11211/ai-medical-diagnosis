"""
Image processing utilities for neural style transfer
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import torch


class ImageProcessor:
    """
    Utility class for image processing operations
    """

    @staticmethod
    def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Load image from file

        Args:
            path: Path to image file
            size: Optional target size (width, height)

        Returns:
            PIL Image
        """
        image = Image.open(path).convert('RGB')

        if size is not None:
            image = image.resize(size, Image.LANCZOS)

        return image

    @staticmethod
    def save_image(image: Union[Image.Image, np.ndarray], path: str, quality: int = 95):
        """
        Save image to file

        Args:
            image: PIL Image or numpy array
            path: Output path
            quality: JPEG quality (1-100)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
            image.save(path, quality=quality, optimize=True)
        else:
            image.save(path)

    @staticmethod
    def resize_image(image: Image.Image, max_size: int = 1024,
                    maintain_aspect: bool = True) -> Image.Image:
        """
        Resize image to maximum dimension

        Args:
            image: Input PIL Image
            max_size: Maximum dimension size
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized image
        """
        if maintain_aspect:
            width, height = image.size
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
        else:
            new_width = new_height = max_size

        return image.resize((new_width, new_height), Image.LANCZOS)

    @staticmethod
    def get_optimal_size(image: Image.Image, target_pixels: int = 1920 * 1080) -> Tuple[int, int]:
        """
        Calculate optimal size maintaining aspect ratio for target pixel count

        Args:
            image: Input image
            target_pixels: Target total pixels

        Returns:
            Tuple of (width, height)
        """
        width, height = image.size
        aspect_ratio = width / height

        # Calculate new dimensions
        new_height = int(np.sqrt(target_pixels / aspect_ratio))
        new_width = int(new_height * aspect_ratio)

        # Ensure dimensions are divisible by 8 (important for some models)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        return new_width, new_height

    @staticmethod
    def numpy_to_pil(array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image

        Args:
            array: Numpy array

        Returns:
            PIL Image
        """
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)

        return Image.fromarray(array)

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array

        Args:
            image: PIL Image

        Returns:
            Numpy array
        """
        return np.array(image)

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array

        Args:
            tensor: PyTorch tensor (C, H, W) or (B, C, H, W)

        Returns:
            Numpy array (H, W, C)
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        array = tensor.cpu().detach().numpy()
        array = np.transpose(array, (1, 2, 0))

        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

        return array

    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """
        Adjust image contrast

        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)

        Returns:
            Adjusted image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_saturation(image: Image.Image, factor: float = 1.0) -> Image.Image:
        """
        Adjust image saturation

        Args:
            image: Input image
            factor: Saturation factor (1.0 = no change)

        Returns:
            Adjusted image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def make_divisible_by(size: Tuple[int, int], divisor: int = 8) -> Tuple[int, int]:
        """
        Adjust dimensions to be divisible by divisor

        Args:
            size: (width, height)
            divisor: Number to divide by

        Returns:
            Adjusted (width, height)
        """
        width, height = size
        width = (width // divisor) * divisor
        height = (height // divisor) * divisor
        return width, height

    @staticmethod
    def create_grid(images: list, grid_size: Optional[Tuple[int, int]] = None,
                   spacing: int = 10) -> Image.Image:
        """
        Create image grid

        Args:
            images: List of PIL Images
            grid_size: Optional (rows, cols), auto-calculated if None
            spacing: Spacing between images in pixels

        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided")

        # Calculate grid size if not provided
        if grid_size is None:
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size

        # Get image dimensions (assume all same size)
        img_width, img_height = images[0].size

        # Calculate grid dimensions
        grid_width = cols * img_width + (cols - 1) * spacing
        grid_height = rows * img_height + (rows - 1) * spacing

        # Create blank grid
        grid = Image.new('RGB', (grid_width, grid_height), color='white')

        # Place images
        for idx, img in enumerate(images):
            if idx >= rows * cols:
                break

            row = idx // cols
            col = idx % cols

            x = col * (img_width + spacing)
            y = row * (img_height + spacing)

            grid.paste(img, (x, y))

        return grid
