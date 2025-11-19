# Audio Noise Cancellation Guide

## Overview

This project includes a comprehensive real-time audio noise cancellation system optimized for Windows 11. It provides low-latency audio processing with multiple noise reduction algorithms suitable for various use cases including:

- **Live Communication**: Video calls, streaming, podcasts
- **Content Creation**: Recording, voice-overs, music production
- **Gaming**: Clear voice chat with minimal background noise
- **Accessibility**: Improved audio clarity for hearing assistance

## Features

### Core Capabilities

- **Real-Time Processing**: Low-latency audio capture and playback
- **Multiple Algorithms**:
  - Spectral Subtraction
  - Wiener Filtering
  - NoiseReduce Library Integration
- **Windows 11 Optimized**: WASAPI support for minimal latency
- **Adaptive Noise Profiling**: Learn and remove specific noise patterns
- **File Processing**: Batch process audio files
- **Live Monitoring**: Real-time audio visualization
- **Configurable**: Extensive configuration options for different use cases

### Performance

- **Latency Modes**:
  - Low: ~10-20ms (gaming, live streaming)
  - Medium: ~20-40ms (general use)
  - High: ~40-80ms (high quality, recording)

- **Sample Rates**: 8kHz to 96kHz
- **Channels**: Mono or Stereo
- **Real-time Factor**: < 0.5x (processes faster than real-time)

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. List Audio Devices

Find your input/output devices:

```bash
python main.py audio-devices
```

### 3. Real-Time Noise Cancellation

Start live audio processing:

```bash
# Basic usage
python main.py audio-live --monitor

# With noise profile
python main.py audio-live --monitor --noise-profile

# Low latency for gaming
python main.py audio-live --monitor --latency-mode low

# High quality for recording
python main.py audio-live --monitor --record --latency-mode high
```

### 4. Process Audio Files

Clean up existing audio files:

```bash
python main.py audio-file --audio-input input.wav --audio-output output.wav

# With noise profile from file
python main.py audio-file --audio-input input.wav --noise-profile
```

## Usage Examples

### Example 1: Video Call Noise Reduction

```bash
python main.py audio-live \
  --monitor \
  --latency-mode low \
  --noise-reduction 0.7 \
  --noise-profile
```

1. Run the command
2. Capture 2 seconds of ambient noise (stay silent)
3. Start talking - background noise is removed in real-time

### Example 2: Podcast Recording

```bash
python main.py audio-live \
  --record \
  --latency-mode high \
  --noise-reduction 0.9 \
  --noise-profile
```

1. Capture noise profile in your recording environment
2. Record your podcast with real-time noise cancellation
3. Output is automatically saved to `output/cleaned_audio_<timestamp>.wav`

### Example 3: Batch File Processing

```bash
# Process multiple files
for file in *.wav; do
  python main.py audio-file \
    --audio-input "$file" \
    --audio-output "cleaned_$file" \
    --noise-reduction 0.8
done
```

## Configuration

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--monitor` | Enable audio playback | False |
| `--record` | Record processed audio | False |
| `--noise-profile` | Capture noise profile | False |
| `--noise-reduction` | Strength (0.0-1.0) | 0.8 |
| `--latency-mode` | low/medium/high | low |
| `--visualize` | Enable visualization | False |

### Python API

```python
from src.audio import AudioProcessor, AudioConfig

# Create configuration
config = AudioConfig(
    sample_rate=48000,
    chunk_size=1024,
    latency_mode='low',
    noise_reduction_strength=0.8,
    use_spectral_subtraction=True,
    use_wiener_filter=True
)

# Create processor
processor = AudioProcessor(config)

# Optional: Capture noise profile
processor.capture_noise_profile(duration=2.0)

# Start real-time processing
processor.start_processing(monitor=True, record=False)

# ... run for some time ...

# Stop processing
processor.stop_processing()

# Get statistics
stats = processor.get_statistics()
print(f"Average latency: {stats['average_latency_ms']:.2f}ms")
```

## Advanced Configuration

### Audio Configuration Parameters

```python
from src.audio import AudioConfig

config = AudioConfig(
    # Device settings
    sample_rate=48000,          # Hz
    chunk_size=1024,            # Samples per chunk
    channels=1,                 # 1=Mono, 2=Stereo
    input_device=None,          # Auto-select
    output_device=None,         # Auto-select

    # Noise cancellation
    noise_reduction_strength=0.8,  # 0.0 to 1.0
    stationary_noise=True,      # For constant noise
    use_spectral_subtraction=True,
    use_wiener_filter=True,

    # Real-time processing
    latency_mode='low',         # 'low', 'medium', 'high'
    buffer_size=4,              # Number of chunks to buffer

    # Spectral processing
    n_fft=2048,                 # FFT size
    hop_length=512,             # Hop length for STFT

    # Wiener filter
    wiener_alpha=0.98,          # Smoothing factor
    wiener_beta=0.02,           # Noise floor

    # Windows 11 specific
    use_wasapi=True,            # Windows Audio Session API
    exclusive_mode=False        # Exclusive device access
)
```

### Use Case Configurations

#### Gaming / Live Streaming
```python
config = AudioConfig(
    latency_mode='low',
    chunk_size=512,
    noise_reduction_strength=0.6,
    use_wasapi=True
)
```

#### Professional Recording
```python
config = AudioConfig(
    sample_rate=96000,
    latency_mode='high',
    chunk_size=2048,
    noise_reduction_strength=0.9,
    n_fft=4096
)
```

#### Voice Chat
```python
config = AudioConfig(
    sample_rate=16000,
    latency_mode='low',
    noise_reduction_strength=0.7,
    stationary_noise=True
)
```

## Algorithms

### 1. Spectral Subtraction

- **Best for**: Stationary background noise (fan, AC, hum)
- **How it works**: Subtracts noise spectrum from signal spectrum
- **Parameters**: `noise_reduction_strength`

### 2. Wiener Filter

- **Best for**: Non-stationary noise (keyboard, mouse clicks)
- **How it works**: Adaptive filter based on signal-to-noise ratio
- **Parameters**: `wiener_alpha`, `wiener_beta`

### 3. Combined Approach

For best results, both algorithms are applied sequentially:
1. Spectral Subtraction removes constant background noise
2. Wiener Filter adapts to remaining dynamic noise

## Performance Optimization

### Reduce Latency

1. Use `latency_mode='low'`
2. Decrease `chunk_size` (512 or 256)
3. Enable WASAPI: `use_wasapi=True`
4. Reduce `noise_reduction_strength`

### Improve Quality

1. Use `latency_mode='high'`
2. Increase `sample_rate` (96000)
3. Increase `n_fft` size (4096)
4. Increase `noise_reduction_strength`

### Balance Performance

Monitor the real-time factor in statistics:
- < 0.5: Excellent, plenty of headroom
- 0.5-0.8: Good, processing efficiently
- 0.8-1.0: Acceptable, near real-time limit
- \> 1.0: Poor, increase chunk size or reduce quality

## Troubleshooting

### Issue: High Latency

**Solutions**:
- Reduce chunk size
- Switch to `latency_mode='low'`
- Enable WASAPI exclusive mode
- Close other audio applications

### Issue: Audio Artifacts/Distortion

**Solutions**:
- Reduce `noise_reduction_strength`
- Increase chunk size
- Capture better noise profile
- Check input levels (avoid clipping)

### Issue: Insufficient Noise Reduction

**Solutions**:
- Increase `noise_reduction_strength`
- Capture noise profile in actual environment
- Use longer noise profile duration
- Enable both algorithms

### Issue: No Audio Devices Found

**Solutions**:
- Check Windows audio settings
- Update audio drivers
- Run `python main.py audio-devices` to list devices
- Specify device manually in config

## File Format Support

### Supported Formats

- **Input**: WAV (8, 16, 32-bit PCM)
- **Output**: WAV (16-bit PCM)

### Converting Other Formats

Use ffmpeg to convert to WAV:

```bash
# MP3 to WAV
ffmpeg -i input.mp3 output.wav

# M4A to WAV
ffmpeg -i input.m4a output.wav

# Any format to 48kHz mono WAV
ffmpeg -i input.* -ar 48000 -ac 1 output.wav
```

## Examples

See the `src/examples/` directory for complete examples:

- `audio_live_example.py` - Real-time processing
- `audio_file_example.py` - File processing
- `audio_config_example.py` - Configuration options

Run examples:

```bash
python src/examples/audio_live_example.py
python src/examples/audio_file_example.py
python src/examples/audio_config_example.py
```

## API Reference

### AudioConfig

Configuration class for audio processing.

```python
config = AudioConfig(
    sample_rate=48000,
    chunk_size=1024,
    # ... see Advanced Configuration
)
```

### AudioCapture

Real-time audio input capture.

```python
from src.audio import AudioCapture, AudioConfig

config = AudioConfig()
capture = AudioCapture(config)

# List devices
devices = capture.list_devices()

# Capture noise profile
noise = capture.capture_noise_profile(duration=2.0)

# Start capture
capture.start_capture()

# Get audio chunk
chunk = capture.get_audio_chunk()

# Stop capture
capture.stop_capture()
```

### NoiseCanceller

Noise cancellation algorithms.

```python
from src.audio import NoiseCanceller, AudioConfig

config = AudioConfig()
canceller = NoiseCanceller(config)

# Set noise profile
canceller.set_noise_profile(noise_audio)

# Process audio chunk
cleaned = canceller.process_chunk(audio_chunk)

# Reset state
canceller.reset()
```

### AudioProcessor

High-level audio processing pipeline.

```python
from src.audio import AudioProcessor, AudioConfig

config = AudioConfig()
processor = AudioProcessor(config)

# List devices
processor.list_devices()

# Real-time processing
processor.start_processing(monitor=True, record=True)
# ... processing runs in background ...
processor.stop_processing()

# File processing
output = processor.process_file('input.wav', 'output.wav')

# Get statistics
stats = processor.get_statistics()
```

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub.

## License

This project is provided as-is for educational and research purposes.
