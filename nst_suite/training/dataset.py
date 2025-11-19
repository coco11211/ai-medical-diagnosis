"""Dataset classes for style transfer training."""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List
import random


class StyleDataset(Dataset):
    """
    Dataset for training fast style transfer models.
    """

    def __init__(
        self,
        content_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize style dataset.

        Args:
            content_dir: Directory containing content images
            image_size: Size to resize images to
            transform: Optional custom transform
        """
        self.content_dir = content_dir
        self.image_size = image_size

        # Get list of image files
        self.image_files = self._get_image_files(content_dir)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {content_dir}")

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files in directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Transformed image tensor
        """
        image_path = self.image_files[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            # If error, return random other image
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class MultiStyleDataset(Dataset):
    """
    Dataset for training multi-style transfer models.
    """

    def __init__(
        self,
        content_dir: str,
        style_images: List[str],
        image_size: int = 256
    ):
        """
        Initialize multi-style dataset.

        Args:
            content_dir: Directory containing content images
            style_images: List of paths to style images
            image_size: Size to resize images to
        """
        self.content_dataset = StyleDataset(content_dir, image_size)
        self.style_images = style_images
        self.num_styles = len(style_images)

        # Load and transform style images
        self.style_tensors = []
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        for style_path in style_images:
            style_image = Image.open(style_path).convert('RGB')
            style_tensor = transform(style_image)
            self.style_tensors.append(style_tensor)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.content_dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (content_image, style_index, style_image)
        """
        content_image = self.content_dataset[idx]

        # Randomly select a style
        style_idx = random.randint(0, self.num_styles - 1)
        style_image = self.style_tensors[style_idx]

        return content_image, style_idx, style_image
