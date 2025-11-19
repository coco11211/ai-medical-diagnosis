"""
Example: Audio Configuration
Demonstrates various audio configuration options
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio import AudioConfig, AudioProcessor


def main():
    """Demonstrate audio configuration options"""

    print("="*70)
    print("Audio Configuration Examples")
    print("="*70)
    print()

    # Example 1: Default configuration
    print("1. Default Configuration:")
    config1 = AudioConfig()
    print(f"   Sample Rate: {config1.sample_rate} Hz")
    print(f"   Chunk Size: {config1.chunk_size}")
    print(f"   Latency Mode: {config1.latency_mode}")
    print(f"   Noise Reduction: {config1.noise_reduction_strength * 100}%")
    print()

    # Example 2: Low-latency configuration
    print("2. Low-Latency Configuration (for gaming, live streaming):")
    config2 = AudioConfig(
        sample_rate=48000,
        latency_mode='low',
        chunk_size=512,
        noise_reduction_strength=0.6,
        use_wasapi=True,
        exclusive_mode=False
    )
    latency = config2.chunk_size / config2.sample_rate * 1000
    print(f"   Sample Rate: {config2.sample_rate} Hz")
    print(f"   Chunk Size: {config2.chunk_size}")
    print(f"   Theoretical Latency: {latency:.2f} ms")
    print(f"   WASAPI: {config2.use_wasapi}")
    print()

    # Example 3: High-quality configuration
    print("3. High-Quality Configuration (for recording, production):")
    config3 = AudioConfig(
        sample_rate=96000,
        latency_mode='high',
        chunk_size=2048,
        noise_reduction_strength=0.9,
        use_spectral_subtraction=True,
        use_wiener_filter=True,
        n_fft=4096
    )
    print(f"   Sample Rate: {config3.sample_rate} Hz")
    print(f"   Chunk Size: {config3.chunk_size}")
    print(f"   FFT Size: {config3.n_fft}")
    print(f"   Noise Reduction: {config3.noise_reduction_strength * 100}%")
    print()

    # Example 4: Custom configuration for specific noise
    print("4. Custom Configuration (stationary noise like fan, AC):")
    config4 = AudioConfig(
        noise_reduction_strength=0.85,
        stationary_noise=True,
        use_spectral_subtraction=True,
        use_wiener_filter=True,
        wiener_alpha=0.95,  # Higher smoothing
        noise_profile_duration=3.0  # Longer noise profile
    )
    print(f"   Stationary Noise Mode: {config4.stationary_noise}")
    print(f"   Noise Profile Duration: {config4.noise_profile_duration}s")
    print(f"   Wiener Alpha: {config4.wiener_alpha}")
    print()

    # Example 5: Validate configuration
    print("5. Configuration Validation:")
    try:
        config5 = AudioConfig(
            sample_rate=48000,
            chunk_size=1024,
            noise_reduction_strength=0.8
        )
        config5.validate()
        print("   Configuration is valid!")
    except ValueError as e:
        print(f"   Configuration error: {e}")
    print()

    # Example 6: Save and load configuration
    print("6. Configuration to Dictionary:")
    config_dict = config1.to_dict()
    print(f"   Keys: {list(config_dict.keys())[:5]}... (showing first 5)")
    print(f"   Total parameters: {len(config_dict)}")
    print()

    # Example 7: Get latency settings
    print("7. Latency Mode Profiles:")
    for mode in ['low', 'medium', 'high']:
        config = AudioConfig(latency_mode=mode)
        settings = config.get_latency_settings()
        latency = settings['chunk_size'] / config.sample_rate * 1000
        print(f"   {mode.upper():8s}: chunk={settings['chunk_size']:4d}, "
              f"buffer={settings['buffer_size']}, latency≈{latency:.2f}ms")
    print()

    print("="*70)
    print("Configuration examples complete!")
    print("\nTip: Use these configurations as templates for your use case")
    print("="*70)


if __name__ == "__main__":
    main()
