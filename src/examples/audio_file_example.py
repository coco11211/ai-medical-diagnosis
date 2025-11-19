"""
Example: Audio File Noise Reduction
Demonstrates batch processing of audio files with noise cancellation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio import AudioProcessor, AudioConfig
from src.audio.audio_utils import AudioFileHandler, AudioMetrics


def main():
    """Run audio file processing example"""

    print("="*70)
    print("Audio File Noise Reduction Example")
    print("="*70)
    print()

    # Example: Process a WAV file
    input_file = "input_audio.wav"  # Replace with your file
    output_file = "output_audio_cleaned.wav"

    # Check if file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        print("\nTo use this example:")
        print("1. Create or provide a WAV audio file")
        print("2. Update the input_file variable with the correct path")
        print("3. Run the script again")
        return

    # Get file information
    print(f"Input file: {input_file}")
    file_info = AudioFileHandler.get_file_info(input_file)
    print(f"  Sample rate: {file_info['sample_rate']} Hz")
    print(f"  Channels: {file_info['channels']}")
    print(f"  Duration: {file_info['duration_s']:.2f} seconds")
    print(f"  Bit depth: {file_info['bit_depth']} bits")
    print()

    # Read audio file
    print("Reading audio file...")
    audio_data, sample_rate, channels = AudioFileHandler.read_wav(input_file)

    # Analyze original audio
    print("\nOriginal Audio Analysis:")
    original_analysis = AudioMetrics.analyze_audio(audio_data, sample_rate)
    print(f"  RMS Level: {original_analysis['rms_db']:.2f} dB")
    print(f"  Peak Level: {original_analysis['peak_db']:.2f} dB")
    print(f"  Crest Factor: {original_analysis['crest_factor_db']:.2f} dB")
    print(f"  Dynamic Range: {original_analysis['dynamic_range_db']:.2f} dB")

    # Create audio configuration
    config = AudioConfig(
        sample_rate=sample_rate,
        channels=channels,
        noise_reduction_strength=0.8,
        use_spectral_subtraction=True,
        use_wiener_filter=True
    )

    # Create audio processor
    processor = AudioProcessor(config)

    # Optional: Use first 2 seconds as noise profile
    print("\nUsing first 2 seconds as noise profile...")
    noise_samples = min(len(audio_data), int(2.0 * sample_rate))
    processor.canceller.set_noise_profile(audio_data[:noise_samples])

    # Process file
    print(f"\nProcessing audio file...")
    output_path = processor.process_file(input_file, output_file)

    # Read processed audio and analyze
    print("\nReading processed audio...")
    processed_data, _, _ = AudioFileHandler.read_wav(output_path)

    print("\nProcessed Audio Analysis:")
    processed_analysis = AudioMetrics.analyze_audio(processed_data, sample_rate)
    print(f"  RMS Level: {processed_analysis['rms_db']:.2f} dB")
    print(f"  Peak Level: {processed_analysis['peak_db']:.2f} dB")
    print(f"  Crest Factor: {processed_analysis['crest_factor_db']:.2f} dB")
    print(f"  Dynamic Range: {processed_analysis['dynamic_range_db']:.2f} dB")

    # Calculate improvement
    noise_reduction = original_analysis['rms_db'] - processed_analysis['rms_db']
    print(f"\nEstimated Noise Reduction: {abs(noise_reduction):.2f} dB")

    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Output saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
