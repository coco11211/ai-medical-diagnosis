"""Image preprocessing utilities for medical imaging."""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import torch
from torchvision import transforms
from skimage import exposure, filters
from loguru import logger


class ImagePreprocessor:
    """Preprocess medical images for deep learning models."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize image preprocessor.

        Args:
            target_size: Target image size
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Define normalization transforms
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        # Define augmentation transforms
        self.augmentation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

        # ImageNet normalization (commonly used for transfer learning)
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        logger.info(f"Initialized ImagePreprocessor with target_size={target_size}")

    def preprocess(
        self,
        image: np.ndarray,
        apply_clahe: bool = False,
        apply_gaussian: bool = False
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image (numpy array)
            apply_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            apply_gaussian: Whether to apply Gaussian blur for noise reduction

        Returns:
            Preprocessed image tensor
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # Apply CLAHE if requested
        if apply_clahe:
            image = self.apply_clahe(image)

        # Apply Gaussian blur if requested
        if apply_gaussian:
            image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply transforms
        if self.augment:
            image_tensor = self.augmentation_transform(image)
        else:
            image_tensor = self.basic_transform(image)

        # Apply normalization
        if self.normalize:
            image_tensor = self.normalize_transform(image_tensor)

        return image_tensor

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE to enhance contrast.

        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            CLAHE-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            image = clahe.apply(image)

        return image

    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization.

        Args:
            image: Input image

        Returns:
            Equalized image
        """
        if len(image.shape) == 3:
            # Apply to each channel
            equalized = np.zeros_like(image)
            for i in range(3):
                equalized[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return equalized
        else:
            return cv2.equalizeHist(image)

    def apply_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge enhancement using unsharp masking.

        Args:
            image: Input image

        Returns:
            Edge-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)

        # Unsharp mask
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        if len(image.shape) == 3:
            # Apply enhancement to all channels
            enhanced = image.copy()
            for i in range(3):
                enhanced[:, :, i] = cv2.addWeighted(image[:, :, i], 1.5,
                                                    cv2.GaussianBlur(image[:, :, i], (0, 0), 3), -0.5, 0)
            return enhanced

        return sharpened

    def batch_preprocess(
        self,
        images: List[np.ndarray],
        apply_clahe: bool = False,
        apply_gaussian: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of input images
            apply_clahe: Whether to apply CLAHE
            apply_gaussian: Whether to apply Gaussian blur

        Returns:
            Batch of preprocessed image tensors
        """
        preprocessed = []

        for image in images:
            tensor = self.preprocess(image, apply_clahe, apply_gaussian)
            preprocessed.append(tensor)

        # Stack into batch
        batch = torch.stack(preprocessed)

        logger.debug(f"Preprocessed batch of {len(images)} images: shape={batch.shape}")

        return batch

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor for visualization.

        Args:
            tensor: Normalized tensor

        Returns:
            Denormalized numpy array
        """
        if self.normalize:
            # Reverse ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean

        # Convert to numpy
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image

    def apply_morphological_operations(
        self,
        image: np.ndarray,
        operation: str = 'close',
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply morphological operations.

        Args:
            image: Input image
            operation: Type of operation ('erode', 'dilate', 'open', 'close')
            kernel_size: Size of morphological kernel

        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if operation == 'erode':
            return cv2.erode(image, kernel)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def remove_noise(
        self,
        image: np.ndarray,
        method: str = 'bilateral'
    ) -> np.ndarray:
        """
        Remove noise from image.

        Args:
            image: Input image
            method: Noise removal method ('bilateral', 'median', 'gaussian')

        Returns:
            Denoised image
        """
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            raise ValueError(f"Unknown method: {method}")
