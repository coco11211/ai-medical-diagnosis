"""
GPU and CUDA Acceleration Utilities
Provides efficient GPU management and optimization for neural style transfer
"""

import torch
import torch.cuda as cuda
from typing import Optional, Dict, List
import logging
import platform
import subprocess


class GPUManager:
    """
    Manages GPU resources and CUDA optimization for style transfer
    """

    def __init__(self):
        """Initialize GPU manager"""
        self.device = self._detect_device()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

        if self.gpu_available:
            self._setup_cuda()

        logging.info(f"GPU Manager initialized - Device: {self.device}")
        logging.info(f"GPUs available: {self.gpu_count}")

    def _detect_device(self) -> str:
        """
        Detect best available device

        Returns:
            Device string ('cuda', 'cuda:0', etc., or 'cpu')
        """
        if torch.cuda.is_available():
            # Check for multiple GPUs and select the one with most memory
            if torch.cuda.device_count() > 1:
                max_memory = 0
                best_device = 0

                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory

                    if total_memory > max_memory:
                        max_memory = total_memory
                        best_device = i

                return f'cuda:{best_device}'
            else:
                return 'cuda'
        else:
            logging.warning("CUDA not available. Using CPU (slower performance expected)")
            return 'cpu'

    def _setup_cuda(self):
        """Setup CUDA optimizations"""
        # Enable cuDNN auto-tuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Enable cuDNN for better performance
        torch.backends.cudnn.enabled = True

        # Set memory allocation strategy (for Windows 11 optimization)
        if platform.system() == 'Windows':
            # Use expandable segments on Windows for better memory management
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Could not optimize CUDA memory: {e}")

    def get_device(self) -> str:
        """
        Get current device

        Returns:
            Device string
        """
        return self.device

    def get_gpu_info(self) -> Dict[str, any]:
        """
        Get detailed GPU information

        Returns:
            Dictionary containing GPU information
        """
        if not self.gpu_available:
            return {
                'available': False,
                'count': 0,
                'devices': []
            }

        devices = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024 ** 3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            }
            devices.append(device_info)

        return {
            'available': True,
            'count': self.gpu_count,
            'devices': devices,
            'cuda_version': torch.version.cuda
        }

    def get_memory_stats(self, device_id: Optional[int] = None) -> Dict[str, float]:
        """
        Get current GPU memory statistics

        Args:
            device_id: GPU device ID (None for current device)

        Returns:
            Dictionary with memory statistics in GB
        """
        if not self.gpu_available:
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0
            }

        if device_id is not None:
            device = f'cuda:{device_id}'
        else:
            device = self.device

        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024 ** 3)
        free = total - allocated

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': free
        }

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logging.info("GPU cache cleared")

    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize model for inference

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        model.eval()

        # Move to GPU
        model = model.to(self.device)

        # Enable optimization flags
        if self.gpu_available:
            # Enable TF32 on Ampere GPUs for better performance
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Set to no grad mode for inference
        for param in model.parameters():
            param.requires_grad = False

        return model

    def enable_mixed_precision(self) -> bool:
        """
        Enable mixed precision training/inference for better performance

        Returns:
            True if enabled successfully
        """
        if self.gpu_available:
            # Check if GPU supports mixed precision
            if torch.cuda.get_device_capability()[0] >= 7:
                return True
            else:
                logging.warning("GPU does not support efficient mixed precision (requires compute capability >= 7.0)")
                return False
        return False

    def get_optimal_batch_size(self, image_size: tuple, model_params: int = 0) -> int:
        """
        Estimate optimal batch size based on available GPU memory

        Args:
            image_size: Tuple of (height, width)
            model_params: Approximate number of model parameters

        Returns:
            Recommended batch size
        """
        if not self.gpu_available:
            return 1

        # Get available memory
        stats = self.get_memory_stats()
        available_gb = stats['free_gb']

        # Rough estimation (can be refined based on actual usage)
        height, width = image_size
        pixels = height * width

        # Estimate memory per image (in GB)
        # Roughly: image_memory + feature_maps + gradients
        memory_per_image = (pixels * 3 * 4) / (1024 ** 3)  # 4 bytes per float32
        memory_per_image *= 10  # Account for intermediate activations

        # Leave some headroom (use 80% of available memory)
        usable_memory = available_gb * 0.8

        batch_size = max(1, int(usable_memory / memory_per_image))

        logging.info(f"Recommended batch size: {batch_size}")
        return batch_size

    def enable_deterministic_mode(self):
        """Enable deterministic mode for reproducibility (may reduce performance)"""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("Deterministic mode enabled")

    def profile_model(self, model: torch.nn.Module, input_shape: tuple) -> Dict[str, float]:
        """
        Profile model performance

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch, channels, height, width)

        Returns:
            Performance metrics
        """
        import time

        model = model.to(self.device).eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        if self.gpu_available:
            torch.cuda.synchronize()

        # Profile
        iterations = 100
        start_time = time.time()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)

        if self.gpu_available:
            torch.cuda.synchronize()

        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations
        fps = 1.0 / avg_time

        return {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time_s': total_time
        }

    def get_cuda_info(self) -> str:
        """
        Get detailed CUDA information as string

        Returns:
            Formatted CUDA information
        """
        if not self.gpu_available:
            return "CUDA not available"

        info = []
        info.append(f"CUDA Available: {self.gpu_available}")
        info.append(f"CUDA Version: {torch.version.cuda}")
        info.append(f"PyTorch Version: {torch.__version__}")
        info.append(f"Number of GPUs: {self.gpu_count}")
        info.append("")

        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            info.append(f"GPU {i}: {props.name}")
            info.append(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            info.append(f"  Compute Capability: {props.major}.{props.minor}")
            info.append(f"  Multi-Processors: {props.multi_processor_count}")

            # Memory stats
            stats = self.get_memory_stats(i)
            info.append(f"  Allocated: {stats['allocated_gb']:.2f} GB")
            info.append(f"  Free: {stats['free_gb']:.2f} GB")
            info.append("")

        return "\n".join(info)


class TensorPool:
    """
    Tensor memory pool for efficient memory reuse in video processing
    """

    def __init__(self, device: str = 'cuda', max_size: int = 100):
        """
        Initialize tensor pool

        Args:
            device: Device to allocate tensors on
            max_size: Maximum number of tensors to cache
        """
        self.device = device
        self.max_size = max_size
        self.pool: List[torch.Tensor] = []

    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor from pool or allocate new one

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Tensor
        """
        # Try to find matching tensor in pool
        for i, tensor in enumerate(self.pool):
            if tensor.shape == shape and tensor.dtype == dtype:
                return self.pool.pop(i)

        # Allocate new tensor
        return torch.empty(shape, dtype=dtype, device=self.device)

    def return_tensor(self, tensor: torch.Tensor):
        """
        Return tensor to pool

        Args:
            tensor: Tensor to return
        """
        if len(self.pool) < self.max_size:
            self.pool.append(tensor)

    def clear(self):
        """Clear the pool"""
        self.pool.clear()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
