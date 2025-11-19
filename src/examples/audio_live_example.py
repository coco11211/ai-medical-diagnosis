"""
Example: Real-Time Audio Noise Cancellation
Demonstrates live audio capture and noise cancellation on Windows 11
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio import AudioProcessor, AudioConfig


def main():
    """Run real-time audio noise cancellation example"""

    print("="*70)
    print("Real-Time Audio Noise Cancellation Example")
    print("="*70)
    print()

    # Create audio configuration
    config = AudioConfig(
        sample_rate=48000,
        chunk_size=1024,
        channels=1,
        noise_reduction_strength=0.8,
        latency_mode='low',
        use_spectral_subtraction=True,
        use_wiener_filter=True,
        monitor_enabled=True
    )

    # Create audio processor
    processor = AudioProcessor(config)

    # List available devices
    print("Available Audio Devices:")
    processor.list_devices()

    # Optional: Capture noise profile
    print("\nCapture noise profile? (y/n): ", end='')
    response = input().lower()

    if response == 'y':
        processor.capture_noise_profile(duration=2.0)

    # Start real-time processing
    print("\nStarting real-time audio processing...")
    print("Speak into your microphone to test noise cancellation")
    print("Press Ctrl+C to stop\n")

    try:
        # Start processing with monitoring enabled
        processor.start_processing(monitor=True, record=False)

        # Keep running until interrupted
        while True:
            time.sleep(0.1)

            # Print statistics every 5 seconds
            stats = processor.get_statistics()
            if stats and stats['frames_processed'] % 50 == 0:
                print(f"\rProcessed: {stats['frames_processed']} frames | "
                      f"Latency: {stats['average_latency_ms']:.2f}ms",
                      end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        processor.stop_processing()

        # Print final statistics
        stats = processor.get_statistics()
        if stats:
            print("\n" + "="*70)
            print("Processing Statistics")
            print("="*70)
            print(f"Total frames processed: {stats['frames_processed']}")
            print(f"Average latency: {stats['average_latency_ms']:.2f} ms")
            print(f"Min latency: {stats['min_latency_ms']:.2f} ms")
            print(f"Max latency: {stats['max_latency_ms']:.2f} ms")
            print(f"Real-time factor: {stats['realtime_factor']:.2f}x")
            print("="*70)


if __name__ == "__main__":
    main()
