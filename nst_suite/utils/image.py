"""Image processing utilities."""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import cv2


def load_image(
    image_path: str,
    max_size: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file
        max_size: Maximum size for the longest dimension
        keep_aspect_ratio: Whether to keep aspect ratio when resizing

    Returns:
        Image as numpy array (H, W, C) in [0, 255]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Resize if needed
    if max_size is not None:
        image = resize_image(image, max_size, keep_aspect_ratio)

    return np.array(image)


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file.

    Args:
        image: Image as numpy array (H, W, C) in [0, 255]
        output_path: Path to save image
    """
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)


def resize_image(
    image: Union[Image.Image, np.ndarray],
    max_size: int,
    keep_aspect_ratio: bool = True
) -> Union[Image.Image, np.ndarray]:
    """
    Resize image to maximum size.

    Args:
        image: PIL Image or numpy array
        max_size: Maximum size for the longest dimension
        keep_aspect_ratio: Whether to keep aspect ratio

    Returns:
        Resized image (same type as input)
    """
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
    else:
        w, h = image.size
        pil_image = image

    if keep_aspect_ratio:
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
    else:
        new_w = new_h = max_size

    resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if is_numpy:
        return np.array(resized)
    return resized


def preprocess_image(
    image: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Preprocess image for neural network.

    Args:
        image: Image as numpy array (H, W, C) in [0, 255]
        device: PyTorch device

    Returns:
        Preprocessed tensor of shape (1, C, H, W) in [0, 1]
    """
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert to tensor and rearrange dimensions
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # Move to device
    return tensor.to(device)


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess tensor to image.

    Args:
        tensor: Tensor of shape (1, C, H, W) in [0, 1]

    Returns:
        Image as numpy array (H, W, C) in [0, 255]
    """
    # Move to CPU and remove batch dimension
    tensor = tensor.cpu().squeeze(0)

    # Clamp to [0, 1] and rearrange dimensions
    image = tensor.clamp(0, 1).permute(1, 2, 0).numpy()

    # Convert to [0, 255]
    image = (image * 255).astype(np.uint8)

    return image


def resize_to_match(
    image: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image to match target shape.

    Args:
        image: Image as numpy array
        target_shape: Target (height, width)

    Returns:
        Resized image
    """
    h, w = target_shape
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 8
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image so dimensions are multiples of given value.

    Args:
        image: Image as numpy array (H, W, C)
        multiple: Value to pad to multiple of

    Returns:
        Padded image and padding amounts (top, bottom, left, right)
    """
    h, w = image.shape[:2]

    # Calculate padding
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad image
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_REFLECT
    )

    return padded, (top, bottom, left, right)


def unpad_image(
    image: np.ndarray,
    padding: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Remove padding from image.

    Args:
        image: Padded image
        padding: Padding amounts (top, bottom, left, right)

    Returns:
        Unpadded image
    """
    top, bottom, left, right = padding
    h, w = image.shape[:2]

    if bottom == 0:
        bottom = h
    else:
        bottom = h - bottom

    if right == 0:
        right = w
    else:
        right = w - right

    return image[top:bottom, left:right]


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Blend two images together.

    Args:
        image1: First image
        image2: Second image
        alpha: Blending factor (0 = all image1, 1 = all image2)

    Returns:
        Blended image
    """
    alpha = np.clip(alpha, 0, 1)
    return (image1 * (1 - alpha) + image2 * alpha).astype(np.uint8)


def convert_color_space(
    image: np.ndarray,
    source: str = 'RGB',
    target: str = 'BGR'
) -> np.ndarray:
    """
    Convert image color space.

    Args:
        image: Input image
        source: Source color space ('RGB', 'BGR', 'HSV', 'LAB')
        target: Target color space

    Returns:
        Converted image
    """
    if source == target:
        return image

    conversion_code = getattr(
        cv2,
        f'COLOR_{source}2{target}',
        None
    )

    if conversion_code is None:
        raise ValueError(f"Unsupported conversion: {source} to {target}")

    return cv2.cvtColor(image, conversion_code)
