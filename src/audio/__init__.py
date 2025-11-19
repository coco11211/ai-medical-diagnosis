"""
Audio Processing Module for Real-Time Noise Cancellation
Supports live audio capture, noise removal, and real-time processing on Windows 11
"""

from .audio_capture import AudioCapture
from .noise_canceller import NoiseCanceller
from .audio_processor import AudioProcessor
from .audio_config import AudioConfig

__all__ = [
    'AudioCapture',
    'NoiseCanceller',
    'AudioProcessor',
    'AudioConfig'
]
