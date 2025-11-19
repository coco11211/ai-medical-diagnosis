"""CUDA utilities for GPU acceleration."""

import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def check_cuda() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_cuda_info() -> Dict[str, any]:
    """
    Get CUDA device information.

    Returns:
        Dictionary with CUDA information
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'device_name': None,
        'compute_capability': None,
        'total_memory_gb': 0,
        'allocated_memory_gb': 0,
        'cached_memory_gb': 0,
    }

    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['compute_capability'] = torch.cuda.get_device_capability(0)

        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)

        info['total_memory_gb'] = total_memory / 1e9
        info['allocated_memory_gb'] = allocated_memory / 1e9
        info['cached_memory_gb'] = cached_memory / 1e9

    return info


def print_cuda_info():
    """Print CUDA information to console."""
    info = get_cuda_info()

    print("\n=== CUDA Information ===")
    print(f"CUDA Available: {info['available']}")

    if info['available']:
        print(f"Device Count: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Compute Capability: {info['compute_capability']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Allocated Memory: {info['allocated_memory_gb']:.2f} GB")
        print(f"Cached Memory: {info['cached_memory_gb']:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")

    print("========================\n")


def optimize_cuda_memory():
    """
    Optimize CUDA memory usage.

    This function performs several optimizations:
    1. Empty CUDA cache
    2. Enable memory efficient settings
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping optimization")
        return

    # Empty cache
    torch.cuda.empty_cache()

    # Enable TF32 for faster computation on Ampere GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuner for faster convolutions
    torch.backends.cudnn.benchmark = True

    logger.info("CUDA memory optimizations applied")


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    max_memory_usage: float = 0.8
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        model: PyTorch model
        input_shape: Shape of a single input (C, H, W)
        max_memory_usage: Maximum fraction of GPU memory to use

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 1

    # Get available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * max_memory_usage

    # Estimate memory per sample (rough estimate)
    # This is a heuristic and may need adjustment
    c, h, w = input_shape
    bytes_per_element = 4  # float32

    # Input memory
    input_memory = c * h * w * bytes_per_element

    # Rough estimate: model forward + backward uses ~3x input memory
    total_per_sample = input_memory * 3

    # Calculate batch size
    batch_size = int(available_memory / total_per_sample)

    # Ensure at least 1
    batch_size = max(1, batch_size)

    # Cap at reasonable maximum
    batch_size = min(batch_size, 32)

    return batch_size


def setup_device(device_id: Optional[int] = None) -> torch.device:
    """
    Setup PyTorch device.

    Args:
        device_id: CUDA device ID. If None, auto-selects best device.

    Returns:
        PyTorch device
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device('cpu')

    if device_id is None:
        # Auto-select device with most free memory
        device_id = get_device_with_most_memory()

    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    logger.info(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")

    return device


def get_device_with_most_memory() -> int:
    """
    Get CUDA device with most free memory.

    Returns:
        Device ID with most free memory
    """
    if not torch.cuda.is_available():
        return 0

    max_free_memory = 0
    best_device = 0

    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        free_memory = props.total_memory - torch.cuda.memory_allocated(device_id)

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = device_id

    return best_device


class CUDAMemoryTracker:
    """Context manager for tracking CUDA memory usage."""

    def __init__(self, name: str = "Operation"):
        """
        Initialize memory tracker.

        Args:
            name: Name of the operation being tracked
        """
        self.name = name
        self.start_allocated = 0
        self.start_cached = 0

    def __enter__(self):
        """Enter context manager."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_cached = torch.cuda.memory_reserved()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and print memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_cached = torch.cuda.memory_reserved()

            allocated_diff = (end_allocated - self.start_allocated) / 1e6
            cached_diff = (end_cached - self.start_cached) / 1e6

            logger.info(
                f"{self.name} - "
                f"Allocated: {allocated_diff:+.2f} MB, "
                f"Cached: {cached_diff:+.2f} MB"
            )
