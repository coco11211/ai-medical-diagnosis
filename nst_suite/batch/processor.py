"""Batch processing for multiple images/videos."""

import os
from pathlib import Path
from typing import List, Optional, Callable
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from ..core.transfer import FastStyleTransfer
from ..utils.image import load_image, save_image
from ..video.processor import VideoStyleTransfer


class BatchProcessor:
    """
    Batch processor for processing multiple images and videos.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize batch processor.

        Args:
            model_path: Path to style transfer model
            device: PyTorch device
            num_workers: Number of parallel workers (None = auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_path = model_path

        # Initialize style transfer
        self.transfer = FastStyleTransfer(model_path, device=self.device)

        if num_workers is None:
            num_workers = min(4, mp.cpu_count())
        self.num_workers = num_workers

    def process_images(
        self,
        input_paths: List[str],
        output_dir: str,
        batch_size: int = 8,
        show_progress: bool = True,
        preserve_structure: bool = False
    ):
        """
        Process multiple images.

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            batch_size: Batch size for GPU processing
            show_progress: Whether to show progress bar
            preserve_structure: Preserve directory structure
        """
        os.makedirs(output_dir, exist_ok=True)

        # Process in batches
        num_images = len(input_paths)
        num_batches = (num_images + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing images")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)

            batch_paths = input_paths[start_idx:end_idx]

            # Load images
            images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = load_image(path)
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if len(images) == 0:
                continue

            # Process batch
            stylized_images = self.transfer.transfer_batch(images)

            # Save images
            for img_path, stylized_img in zip(valid_paths, stylized_images):
                # Determine output path
                if preserve_structure:
                    # Preserve directory structure
                    rel_path = os.path.relpath(img_path)
                    output_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                else:
                    # Flat output directory
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_dir, filename)

                # Add suffix to avoid overwriting
                name, ext = os.path.splitext(output_path)
                output_path = f"{name}_stylized{ext}"

                save_image(stylized_img, output_path)

        print(f"Processed {num_images} images. Output saved to {output_dir}")

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Whether to search recursively
            extensions: List of file extensions to process
            **kwargs: Additional arguments for process_images
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Find all images
        image_paths = []

        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(input_dir, file))

        print(f"Found {len(image_paths)} images in {input_dir}")

        if len(image_paths) == 0:
            print("No images found!")
            return

        # Process images
        self.process_images(image_paths, output_dir, **kwargs)

    def process_videos(
        self,
        input_paths: List[str],
        output_dir: str,
        batch_size: int = 4,
        show_progress: bool = True
    ):
        """
        Process multiple videos.

        Args:
            input_paths: List of input video paths
            output_dir: Output directory
            batch_size: Batch size for frame processing
            show_progress: Whether to show progress bar
        """
        os.makedirs(output_dir, exist_ok=True)

        video_processor = VideoStyleTransfer(
            model_path=self.model_path,
            device=self.device
        )

        iterator = input_paths
        if show_progress:
            iterator = tqdm(iterator, desc="Processing videos")

        for video_path in iterator:
            try:
                # Determine output path
                filename = os.path.basename(video_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_stylized{ext}")

                # Process video
                video_processor.process_video(
                    video_path,
                    output_path,
                    batch_size=batch_size,
                    show_progress=False
                )

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        print(f"Processed {len(input_paths)} videos. Output saved to {output_dir}")

    def process_parallel(
        self,
        input_paths: List[str],
        output_dir: str,
        process_fn: Callable[[str, str], None],
        max_workers: Optional[int] = None
    ):
        """
        Process files in parallel using multiple threads.

        Args:
            input_paths: List of input paths
            output_dir: Output directory
            process_fn: Function to process each file (input_path, output_path)
            max_workers: Maximum number of worker threads
        """
        os.makedirs(output_dir, exist_ok=True)

        if max_workers is None:
            max_workers = self.num_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for input_path in input_paths:
                filename = os.path.basename(input_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_stylized{ext}")

                future = executor.submit(process_fn, input_path, output_path)
                futures.append(future)

            # Wait for completion with progress bar
            for future in tqdm(futures, desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")


class MultiStyleBatchProcessor:
    """
    Batch processor for applying multiple styles to images.
    """

    def __init__(
        self,
        model_paths: List[str],
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-style batch processor.

        Args:
            model_paths: List of paths to style models
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
            self.models.append(model)

        self.num_styles = len(self.models)

    def process_with_all_styles(
        self,
        input_path: str,
        output_dir: str,
        style_names: Optional[List[str]] = None
    ):
        """
        Apply all styles to a single image.

        Args:
            input_path: Input image path
            output_dir: Output directory
            style_names: Optional names for each style
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load image
        image = load_image(input_path)

        # Get base filename
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        # Apply each style
        for i, model in enumerate(tqdm(self.models, desc="Applying styles")):
            # Process with style
            stylized = model.transfer(image)

            # Determine output name
            if style_names and i < len(style_names):
                style_name = style_names[i]
            else:
                style_name = f"style_{i}"

            output_path = os.path.join(output_dir, f"{name}_{style_name}{ext}")
            save_image(stylized, output_path)

        print(f"Applied {self.num_styles} styles. Output saved to {output_dir}")

    def create_style_grid(
        self,
        input_paths: List[str],
        output_path: str,
        grid_size: Optional[tuple] = None
    ):
        """
        Create a grid showing multiple images with multiple styles.

        Args:
            input_paths: List of input image paths
            output_path: Output image path
            grid_size: Optional (rows, cols) for grid layout
        """
        import numpy as np
        from math import ceil, sqrt

        # Load images
        images = [load_image(path, max_size=512) for path in input_paths]

        num_images = len(images)
        num_styles = self.num_styles

        # Determine grid size
        if grid_size is None:
            cols = num_styles
            rows = num_images
        else:
            rows, cols = grid_size

        # Apply styles to all images
        all_stylized = []

        for img in tqdm(images, desc="Processing images"):
            row_images = []
            for model in self.models:
                stylized = model.transfer(img)
                row_images.append(stylized)
            all_stylized.append(row_images)

        # Create grid
        # Resize all images to same size
        target_size = (256, 256)
        grid = []

        for row_idx in range(rows):
            if row_idx >= len(all_stylized):
                break

            row = []
            for col_idx in range(cols):
                if col_idx >= len(all_stylized[row_idx]):
                    # Empty cell
                    row.append(np.zeros((*target_size, 3), dtype=np.uint8))
                else:
                    import cv2
                    resized = cv2.resize(
                        all_stylized[row_idx][col_idx],
                        target_size[::-1]
                    )
                    row.append(resized)

            grid.append(np.hstack(row))

        final_grid = np.vstack(grid)
        save_image(final_grid, output_path)

        print(f"Style grid saved to {output_path}")
