"""Unit tests for image processing modules."""

import pytest
import numpy as np
import torch
from src.imaging.image_preprocessor import ImagePreprocessor
from src.imaging.dicom_processor import DICOMProcessor


class TestImagePreprocessor:
    """Test ImagePreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True,
            augment=False
        )
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.normalize is True
        assert preprocessor.augment is False

    def test_preprocess_shape(self, sample_image):
        """Test that preprocessing produces correct shape."""
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        result = preprocessor.preprocess(sample_image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_batch_preprocess(self, sample_image):
        """Test batch preprocessing."""
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        images = [sample_image] * 5

        batch = preprocessor.batch_preprocess(images)

        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (5, 3, 224, 224)

    def test_apply_clahe(self, sample_image):
        """Test CLAHE enhancement."""
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.apply_clahe(sample_image)

        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8

    def test_denormalize(self):
        """Test denormalization."""
        preprocessor = ImagePreprocessor(normalize=True)
        tensor = torch.randn(3, 224, 224)

        denormalized = preprocessor.denormalize(tensor)

        assert isinstance(denormalized, np.ndarray)
        assert denormalized.shape == (224, 224, 3)
        assert denormalized.dtype == np.uint8


class TestDICOMProcessor:
    """Test DICOMProcessor class."""

    def test_initialization(self):
        """Test DICOM processor initialization."""
        processor = DICOMProcessor(
            window_center=40,
            window_width=400,
            target_size=(512, 512)
        )

        assert processor.window_center == 40
        assert processor.window_width == 400
        assert processor.target_size == (512, 512)

    def test_apply_windowing(self):
        """Test intensity windowing."""
        processor = DICOMProcessor()
        pixel_array = np.random.randint(-1000, 1000, (512, 512)).astype(np.float32)

        windowed = processor.apply_windowing(pixel_array)

        assert windowed.dtype == np.uint8
        assert windowed.min() >= 0
        assert windowed.max() <= 255

    def test_resize_image(self, sample_image):
        """Test image resizing."""
        processor = DICOMProcessor(target_size=(256, 256))

        # Convert to grayscale for testing
        gray_image = sample_image[:, :, 0]
        resized = processor.resize_image(gray_image, target_size=(256, 256))

        assert resized.shape == (256, 256)

    def test_normalize_image(self, sample_image):
        """Test image normalization."""
        processor = DICOMProcessor()
        normalized = processor.normalize_image(sample_image)

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_convert_to_3channel(self):
        """Test grayscale to RGB conversion."""
        processor = DICOMProcessor()
        gray_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

        rgb_image = processor.convert_to_3channel(gray_image)

        assert rgb_image.shape == (512, 512, 3)
