"""
Audio Configuration Module
Manages audio processing parameters and device settings
"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class AudioConfig:
    """Configuration for audio processing"""

    # Audio device settings
    sample_rate: int = 48000  # Hz (Windows 11 default high quality)
    chunk_size: int = 2048  # Samples per chunk
    channels: int = 1  # Mono for noise cancellation
    format: str = 'int16'  # Audio format
    input_device: Optional[int] = None  # Auto-select if None
    output_device: Optional[int] = None  # Auto-select if None

    # Noise cancellation settings
    noise_reduction_strength: float = 0.8  # 0.0 to 1.0
    stationary_noise: bool = True  # Stationary vs non-stationary noise
    use_spectral_subtraction: bool = True
    use_wiener_filter: bool = True
    use_rnnoise: bool = False  # Requires additional setup

    # Real-time processing
    latency_mode: str = 'low'  # 'low', 'medium', 'high'
    buffer_size: int = 4  # Number of chunks to buffer

    # Spectral processing
    n_fft: int = 2048  # FFT size
    hop_length: int = 512  # Hop length for STFT
    noise_profile_duration: float = 2.0  # seconds

    # Wiener filter
    wiener_alpha: float = 0.98  # Smoothing factor
    wiener_beta: float = 0.02  # Noise floor

    # Output settings
    save_output: bool = False
    output_path: str = "output/cleaned_audio"
    monitor_enabled: bool = True

    # Windows 11 specific
    use_wasapi: bool = True  # Windows Audio Session API
    exclusive_mode: bool = False  # Exclusive mode for lower latency

    @classmethod
    def from_yaml(cls, config_path: str) -> 'AudioConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        audio_config = config_data.get('audio', {})
        return cls(**audio_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'channels': self.channels,
            'format': self.format,
            'input_device': self.input_device,
            'output_device': self.output_device,
            'noise_reduction_strength': self.noise_reduction_strength,
            'stationary_noise': self.stationary_noise,
            'use_spectral_subtraction': self.use_spectral_subtraction,
            'use_wiener_filter': self.use_wiener_filter,
            'use_rnnoise': self.use_rnnoise,
            'latency_mode': self.latency_mode,
            'buffer_size': self.buffer_size,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'noise_profile_duration': self.noise_profile_duration,
            'wiener_alpha': self.wiener_alpha,
            'wiener_beta': self.wiener_beta,
            'save_output': self.save_output,
            'output_path': self.output_path,
            'monitor_enabled': self.monitor_enabled,
            'use_wasapi': self.use_wasapi,
            'exclusive_mode': self.exclusive_mode
        }

    def get_latency_settings(self) -> Dict[str, int]:
        """Get latency-optimized settings based on latency mode"""
        latency_profiles = {
            'low': {'chunk_size': 512, 'buffer_size': 2},
            'medium': {'chunk_size': 1024, 'buffer_size': 4},
            'high': {'chunk_size': 2048, 'buffer_size': 8}
        }
        return latency_profiles.get(self.latency_mode, latency_profiles['medium'])

    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.chunk_size <= 0 or self.chunk_size > 8192:
            raise ValueError("Chunk size must be between 1 and 8192")
        if self.channels not in [1, 2]:
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError("Noise reduction strength must be between 0.0 and 1.0")
        if self.latency_mode not in ['low', 'medium', 'high']:
            raise ValueError("Latency mode must be 'low', 'medium', or 'high'")
        return True
