"""
Batch Processing for Neural Style Transfer
Efficient processing of multiple images with parallel execution
"""

import torch
import torch.multiprocessing as mp
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict, Callable
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from ..models.resnet_model import ResNetStyleTransfer
from ..utils.gpu_utils import GPUManager
from ..utils.image_utils import ImageProcessor


class BatchProcessor:
    """
    Batch processing for neural style transfer with multi-GPU support
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 num_workers: int = 4):
        """
        Initialize batch processor

        Args:
            model_path: Path to pre-trained model
            device: Device to use (auto-detected if None)
            num_workers: Number of parallel workers
        """
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_device()
        self.num_workers = num_workers

        # Initialize model
        self.model = ResNetStyleTransfer(device=self.device)

        if model_path:
            self.load_model(model_path)

        self.model = self.gpu_manager.optimize_for_inference(self.model)

        self.image_processor = ImageProcessor()

        logging.info(f"Batch processor initialized: {self.num_workers} workers on {self.device}")

    def load_model(self, model_path: str):
        """Load pre-trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        logging.info(f"Model loaded from {model_path}")

    def process_images(self,
                      input_paths: List[str],
                      output_dir: str,
                      size: Optional[tuple] = None,
                      quality: int = 95,
                      preserve_names: bool = True,
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process multiple images in batch

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            size: Optional target size (width, height)
            quality: Output image quality (1-100)
            preserve_names: Keep original filenames
            progress_callback: Optional progress callback

        Returns:
            Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'total': len(input_paths),
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'failed_files': []
        }

        with tqdm(total=len(input_paths), desc="Batch processing") as pbar:
            for idx, input_path in enumerate(input_paths):
                try:
                    start_time = time.time()

                    # Load image
                    image = self.image_processor.load_image(input_path, size)

                    # Process
                    styled_image = self.model.stylize_image(image)

                    # Save
                    if preserve_names:
                        output_filename = Path(input_path).name
                    else:
                        ext = Path(input_path).suffix
                        output_filename = f"styled_{idx:04d}{ext}"

                    output_file = output_path / output_filename
                    self.image_processor.save_image(styled_image, str(output_file), quality)

                    # Track time
                    process_time = time.time() - start_time
                    results['processing_times'].append(process_time)
                    results['successful'] += 1

                except Exception as e:
                    logging.error(f"Failed to process {input_path}: {e}")
                    results['failed'] += 1
                    results['failed_files'].append(input_path)

                pbar.update(1)

                if progress_callback:
                    progress_callback(idx + 1, len(input_paths))

        # Calculate statistics
        if results['processing_times']:
            import numpy as np
            results['avg_time'] = np.mean(results['processing_times'])
            results['total_time'] = sum(results['processing_times'])

        logging.info(f"Batch processing complete: {results['successful']}/{results['total']} successful")

        return results

    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         extensions: List[str] = None,
                         **kwargs) -> Dict:
        """
        Process all images in a directory

        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: List of file extensions to process (default: common image formats)
            **kwargs: Additional arguments for process_images

        Returns:
            Processing statistics
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        # Find all images
        input_path = Path(input_dir)
        image_files = []

        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        image_paths = [str(f) for f in image_files]

        logging.info(f"Found {len(image_paths)} images in {input_dir}")

        return self.process_images(image_paths, output_dir, **kwargs)

    def process_batch_parallel(self,
                              input_paths: List[str],
                              output_dir: str,
                              batch_size: int = 8,
                              **kwargs) -> Dict:
        """
        Process images in parallel batches for better GPU utilization

        Args:
            input_paths: List of input paths
            output_dir: Output directory
            batch_size: Number of images to process simultaneously
            **kwargs: Additional arguments

        Returns:
            Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Split into batches
        batches = [input_paths[i:i + batch_size]
                  for i in range(0, len(input_paths), batch_size)]

        results = {
            'total': len(input_paths),
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'failed_files': []
        }

        with tqdm(total=len(input_paths), desc="Parallel batch processing") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    # Load batch
                    images = []
                    valid_paths = []

                    for path in batch:
                        try:
                            img = self.image_processor.load_image(path)
                            images.append(img)
                            valid_paths.append(path)
                        except Exception as e:
                            logging.error(f"Failed to load {path}: {e}")
                            results['failed'] += 1
                            results['failed_files'].append(path)

                    if not images:
                        continue

                    # Process batch
                    start_time = time.time()
                    styled_images = self._process_batch(images)
                    batch_time = time.time() - start_time

                    # Save results
                    for img, path in zip(styled_images, valid_paths):
                        output_file = output_path / Path(path).name
                        self.image_processor.save_image(img, str(output_file))
                        results['successful'] += 1

                    results['processing_times'].append(batch_time / len(images))

                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    results['failed'] += len(batch)

                pbar.update(len(batch))

        # Calculate statistics
        if results['processing_times']:
            import numpy as np
            results['avg_time'] = np.mean(results['processing_times'])
            results['total_time'] = sum(results['processing_times'])

        return results

    def _process_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process a batch of images simultaneously

        Args:
            images: List of PIL Images

        Returns:
            List of styled PIL Images
        """
        # Convert to tensors
        from torchvision import transforms

        transform = transforms.ToTensor()
        tensors = [transform(img) for img in images]

        # Pad to same size if necessary
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)

        padded_tensors = []
        for t in tensors:
            _, h, w = t.shape
            pad_h = max_h - h
            pad_w = max_w - w

            if pad_h > 0 or pad_w > 0:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))

            padded_tensors.append(t)

        # Stack into batch
        batch = torch.stack(padded_tensors).to(self.device)

        # Process
        with torch.no_grad():
            styled_batch = self.model(batch)

        # Convert back to images
        styled_images = []

        for i, (styled_tensor, original_img) in enumerate(zip(styled_batch, images)):
            # Crop to original size
            w, h = original_img.size
            styled_tensor = styled_tensor[:, :h, :w]

            # Convert to PIL
            styled_tensor = styled_tensor.clamp(0, 1)
            styled_array = styled_tensor.cpu().numpy()
            styled_array = (styled_array.transpose(1, 2, 0) * 255).astype('uint8')
            styled_img = Image.fromarray(styled_array)

            styled_images.append(styled_img)

        return styled_images

    def process_with_multiple_gpus(self,
                                   input_paths: List[str],
                                   output_dir: str,
                                   gpu_ids: List[int] = None) -> Dict:
        """
        Process images using multiple GPUs

        Args:
            input_paths: List of input paths
            output_dir: Output directory
            gpu_ids: List of GPU IDs to use (all available if None)

        Returns:
            Processing statistics
        """
        if gpu_ids is None:
            gpu_ids = list(range(self.gpu_manager.gpu_count))

        if not gpu_ids or self.gpu_manager.gpu_count == 0:
            logging.warning("No GPUs available, falling back to single device")
            return self.process_images(input_paths, output_dir)

        # Split work across GPUs
        chunks = [input_paths[i::len(gpu_ids)] for i in range(len(gpu_ids))]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []

            for gpu_id, chunk in zip(gpu_ids, chunks):
                future = executor.submit(
                    self._process_on_gpu,
                    chunk,
                    output_dir,
                    gpu_id
                )
                futures.append(future)

            # Collect results
            results = {
                'total': len(input_paths),
                'successful': 0,
                'failed': 0,
                'processing_times': [],
                'failed_files': []
            }

            for future in futures:
                gpu_result = future.result()
                results['successful'] += gpu_result['successful']
                results['failed'] += gpu_result['failed']
                results['processing_times'].extend(gpu_result['processing_times'])
                results['failed_files'].extend(gpu_result['failed_files'])

        return results

    @staticmethod
    def _process_on_gpu(input_paths: List[str], output_dir: str, gpu_id: int) -> Dict:
        """Worker function for multi-GPU processing"""
        device = f'cuda:{gpu_id}'
        processor = BatchProcessor(device=device, num_workers=1)
        return processor.process_images(input_paths, output_dir)


class SmartBatchProcessor:
    """
    Intelligent batch processor with auto-optimization
    """

    def __init__(self):
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_device()

    def auto_process(self, input_dir: str, output_dir: str) -> Dict:
        """
        Automatically optimize and process images

        Args:
            input_dir: Input directory
            output_dir: Output directory

        Returns:
            Processing statistics
        """
        # Determine optimal batch size
        batch_size = self.gpu_manager.get_optimal_batch_size((1080, 1920))

        # Create processor
        processor = BatchProcessor(device=self.device)

        # Process with optimal settings
        return processor.process_directory(
            input_dir,
            output_dir,
            batch_size=batch_size
        )
