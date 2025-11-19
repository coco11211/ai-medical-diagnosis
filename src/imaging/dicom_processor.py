"""DICOM image processing and handling."""

import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Union, List
import cv2
from loguru import logger


class DICOMProcessor:
    """Process and handle DICOM medical images."""

    def __init__(
        self,
        window_center: int = 40,
        window_width: int = 400,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize DICOM processor.

        Args:
            window_center: Window center for intensity windowing
            window_width: Window width for intensity windowing
            target_size: Target size for resizing images
        """
        self.window_center = window_center
        self.window_width = window_width
        self.target_size = target_size
        logger.info(f"Initialized DICOMProcessor with window: {window_center}±{window_width}")

    def read_dicom(self, file_path: Union[str, Path]) -> pydicom.Dataset:
        """
        Read DICOM file.

        Args:
            file_path: Path to DICOM file

        Returns:
            DICOM dataset
        """
        try:
            dicom_data = pydicom.dcmread(file_path)
            logger.debug(f"Successfully read DICOM file: {file_path}")
            return dicom_data
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            raise

    def get_pixel_array(self, dicom_data: pydicom.Dataset) -> np.ndarray:
        """
        Extract pixel array from DICOM dataset.

        Args:
            dicom_data: DICOM dataset

        Returns:
            Pixel array
        """
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        # Apply rescale slope and intercept if available
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept

        return pixel_array

    def apply_windowing(
        self,
        pixel_array: np.ndarray,
        window_center: Optional[int] = None,
        window_width: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply intensity windowing to pixel array.

        Args:
            pixel_array: Input pixel array
            window_center: Window center (uses default if None)
            window_width: Window width (uses default if None)

        Returns:
            Windowed pixel array
        """
        wc = window_center if window_center is not None else self.window_center
        ww = window_width if window_width is not None else self.window_width

        # Calculate window boundaries
        lower = wc - (ww / 2)
        upper = wc + (ww / 2)

        # Apply windowing
        windowed = np.clip(pixel_array, lower, upper)

        # Normalize to 0-255
        windowed = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)

        return windowed

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            target_size: Target size (uses default if None)

        Returns:
            Resized image
        """
        size = target_size if target_size is not None else self.target_size
        resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        return resized

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized

    def process_dicom_file(
        self,
        file_path: Union[str, Path],
        apply_window: bool = True,
        resize: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, pydicom.Dataset]:
        """
        Complete DICOM processing pipeline.

        Args:
            file_path: Path to DICOM file
            apply_window: Whether to apply windowing
            resize: Whether to resize image
            normalize: Whether to normalize image

        Returns:
            Tuple of (processed image, DICOM dataset)
        """
        # Read DICOM file
        dicom_data = self.read_dicom(file_path)

        # Get pixel array
        pixel_array = self.get_pixel_array(dicom_data)

        # Apply windowing
        if apply_window:
            pixel_array = self.apply_windowing(pixel_array)

        # Resize
        if resize:
            pixel_array = self.resize_image(pixel_array)

        # Normalize
        if normalize:
            pixel_array = self.normalize_image(pixel_array)

        logger.info(f"Processed DICOM file {file_path}: shape={pixel_array.shape}")

        return pixel_array, dicom_data

    def extract_metadata(self, dicom_data: pydicom.Dataset) -> dict:
        """
        Extract relevant metadata from DICOM dataset.

        Args:
            dicom_data: DICOM dataset

        Returns:
            Dictionary of metadata
        """
        metadata = {
            'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(dicom_data, 'PatientName', 'Unknown')),
            'patient_age': getattr(dicom_data, 'PatientAge', 'Unknown'),
            'patient_sex': getattr(dicom_data, 'PatientSex', 'Unknown'),
            'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
            'modality': getattr(dicom_data, 'Modality', 'Unknown'),
            'body_part': getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
            'study_description': getattr(dicom_data, 'StudyDescription', 'Unknown'),
            'series_description': getattr(dicom_data, 'SeriesDescription', 'Unknown'),
            'rows': int(getattr(dicom_data, 'Rows', 0)),
            'columns': int(getattr(dicom_data, 'Columns', 0)),
        }

        return metadata

    def process_dicom_series(
        self,
        directory: Union[str, Path]
    ) -> Tuple[List[np.ndarray], List[pydicom.Dataset]]:
        """
        Process all DICOM files in a directory (series).

        Args:
            directory: Directory containing DICOM files

        Returns:
            Tuple of (list of processed images, list of DICOM datasets)
        """
        directory = Path(directory)
        dicom_files = sorted(directory.glob("*.dcm"))

        images = []
        datasets = []

        for file_path in dicom_files:
            try:
                image, dicom_data = self.process_dicom_file(file_path)
                images.append(image)
                datasets.append(dicom_data)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        logger.info(f"Processed {len(images)} DICOM files from {directory}")

        return images, datasets

    def convert_to_3channel(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to 3-channel RGB format.

        Args:
            image: Grayscale image

        Returns:
            3-channel RGB image
        """
        if len(image.shape) == 2:
            return np.stack([image, image, image], axis=-1)
        return image
