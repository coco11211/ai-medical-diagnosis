# AI Music Composer

A comprehensive AI-powered music composition system for Windows 11 (and other platforms) featuring LSTM and Transformer models for multi-genre, mood-based music generation.

## Features

### Core Capabilities
- **Multiple AI Models**: LSTM and Transformer architectures for music generation
- **Multi-Genre Support**: Classical, Jazz, Pop, Rock, Electronic, Blues, Ambient, Folk
- **Mood-Based Composition**: Generate music based on 8 different moods (Happy, Sad, Energetic, Calm, Mysterious, Romantic, Aggressive, Peaceful)
- **Harmony & Melody Synthesis**: Automatic harmony generation with chord progressions
- **Real-Time Playback**: Play generated music instantly
- **Multiple Export Formats**: MIDI, WAV, MP3
- **Custom Training**: Train models on your own MIDI files
- **Professional GUI**: User-friendly Windows 11 compatible interface

### Music Theory Integration
- Automatic scale selection based on genre and mood
- Chord progression generation
- Rhythm patterns specific to each genre
- Harmonic and melodic constraints
- Multi-track composition (melody, harmony, drums)

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows 11 (also works on macOS and Linux)

### Required System Packages

**Windows:**
```bash
# Optional: For better audio synthesis
# Download and install FluidSynth from https://github.com/FluidSynth/fluidsynth/releases
# Download and install FFmpeg from https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get install fluidsynth ffmpeg
```

**macOS:**
```bash
brew install fluid-synth ffmpeg
```

### Python Dependencies

Install required packages:
```bash
pip install -r music_composer_requirements.txt
```

## Quick Start

### Running the Application

```bash
python music_composer_app.py
```

This will launch the GUI application with three main tabs:
1. **Compose**: Generate and export music
2. **Train Model**: Train custom models on your data
3. **Settings**: View system information

### Generating Your First Composition

1. Select a **Genre** (e.g., Pop, Jazz, Classical)
2. Choose a **Mood** (e.g., Happy, Calm, Energetic)
3. Set **Duration** (in seconds)
4. Adjust **Tempo** (BPM)
5. Tune **Creativity** slider (temperature)
6. Enable **Add Harmony** and/or **Add Drums**
7. Click **Generate Music**
8. Use **Play MIDI** to preview
9. Export to MIDI, WAV, or MP3

## Usage Guide

### Composition Tab

**Parameters:**
- **Genre**: Musical style (affects scale, rhythm, instrumentation)
- **Mood**: Emotional character (affects tempo, brightness, chord types)
- **Duration**: Length of composition in seconds (10-300s)
- **Tempo**: Speed in BPM (40-200)
- **Creativity**: Sampling temperature (0.5 = conservative, 1.5 = experimental)

**Options:**
- **Add Harmony**: Generates accompanying harmony track
- **Add Drums**: Adds genre-appropriate drum patterns

**Controls:**
- **Generate Music**: Start composition
- **Stop Generation**: Cancel current generation
- **Play MIDI**: Preview generated music
- **Stop**: Stop playback
- **Export MIDI**: Save as MIDI file
- **Export WAV**: Convert and save as WAV
- **Export MP3**: Convert and save as MP3

### Training Tab

**Custom Model Training:**

1. **Select Model Type**: Choose LSTM or Transformer
2. **Select MIDI Files**: Load your training data
3. **Configure Parameters**:
   - Epochs: Number of training iterations (1-500)
   - Batch Size: Samples per batch (8-128)
   - Learning Rate: Optimization rate (default: 0.001)
4. **Start Training**: Begin training process
5. **Save Model**: Save trained model for later use
6. **Load Model**: Load previously saved model

**Training Tips:**
- Use at least 10-20 MIDI files for decent results
- More data = better quality
- Training can take hours depending on data size and epochs
- Models are automatically saved to `checkpoints/` directory
- Data augmentation is applied automatically (transposition, etc.)

## Architecture

### Models

**Transformer Model:**
- Multi-head self-attention mechanism
- Positional encoding
- Configurable layers and attention heads
- Supports conditional generation (genre + mood)
- Better for complex, long-range dependencies

**LSTM Model:**
- Recurrent neural network architecture
- Stacked LSTM layers with dropout
- Embedding layers for notes and conditions
- Better for sequential patterns
- Lower memory requirements

### Music Theory Engine

**Scales:**
- Major, Minor, Dorian, Phrygian, Lydian, Mixolydian
- Harmonic Minor, Melodic Minor
- Pentatonic (Major/Minor)
- Blues, Chromatic

**Chord Progressions:**
- Genre-specific progressions
- Automatic roman numeral analysis
- Support for extended chords (7th, 6th, sus)

**Rhythm Patterns:**
- Genre-specific timing
- Dynamic velocity ranges
- Swing and straight feels

### Export Pipeline

**MIDI Export:**
- Multi-track MIDI files
- Configurable instruments
- Tempo and time signature control

**Audio Synthesis:**
- FluidSynth integration (if available)
- Fallback to basic additive synthesis
- ADSR envelope shaping
- Multiple harmonics

**Format Conversion:**
- MIDI → WAV (44.1kHz, 16-bit)
- WAV → MP3 (configurable bitrate)

## Advanced Usage

### Command Line API

You can also use the system programmatically:

```python
import torch
from music_composer.models.transformer_model import ConditionalTransformer
from music_composer.utils.composition_engine import CompositionEngine, MoodBasedComposer

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConditionalTransformer().to(device)

# Create composition engine
engine = CompositionEngine(model, 'transformer', device)
composer = MoodBasedComposer(engine)

# Generate composition
composition = composer.compose_for_mood(
    mood='happy',
    duration=60,
    genre='pop',
    tempo=120
)

# Save to MIDI
engine.save_composition(composition, 'output.mid')
```

### Training Custom Models

```python
from music_composer.training.trainer import MusicTrainer, DataPreprocessor
from music_composer.models.transformer_model import ConditionalTransformer
from torch.utils.data import DataLoader

# Load data
preprocessor = DataPreprocessor()
sequences = preprocessor.load_midi_files(['file1.mid', 'file2.mid'])

# Prepare datasets
train_dataset, val_dataset = preprocessor.prepare_dataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create and train model
model = ConditionalTransformer()
trainer = MusicTrainer(model, 'transformer', 'cpu', learning_rate=0.001)
trainer.train(train_loader, val_loader, num_epochs=50, conditional=True)
```

## Project Structure

```
music_composer/
├── models/
│   ├── lstm_model.py          # LSTM architecture
│   └── transformer_model.py   # Transformer architecture
├── utils/
│   ├── music_theory.py        # Music theory utilities
│   ├── midi_processor.py      # MIDI processing
│   └── composition_engine.py  # Main composition logic
├── export/
│   └── audio_export.py        # Audio export and playback
├── training/
│   └── trainer.py             # Training pipeline
└── gui/
    └── main_window.py         # GUI application

music_composer_app.py          # Main entry point
music_composer_requirements.txt # Dependencies
```

## Supported Genres

1. **Classical**: Complex harmonies, varied dynamics, traditional forms
2. **Jazz**: Complex chords, swing rhythms, improvisation-style
3. **Pop**: Simple progressions, catchy melodies, moderate tempo
4. **Rock**: Power chords, driving rhythms, pentatonic scales
5. **Electronic**: Repetitive patterns, fast tempo, synthetic feel
6. **Blues**: 12-bar progressions, blues scale, expressive bends
7. **Ambient**: Slow tempo, atmospheric, minimal percussion
8. **Folk**: Simple melodies, acoustic feel, traditional progressions

## Supported Moods

1. **Happy**: Major keys, bright, upbeat tempo
2. **Sad**: Minor keys, slower tempo, lower dynamics
3. **Energetic**: Fast tempo, driving rhythm, high energy
4. **Calm**: Slow tempo, smooth transitions, gentle dynamics
5. **Mysterious**: Modal scales, unusual progressions, moderate tempo
6. **Romantic**: Lush harmonies, flowing melodies, moderate tempo
7. **Aggressive**: Dissonant harmonies, fast tempo, high intensity
8. **Peaceful**: Consonant harmonies, slow tempo, soft dynamics

## Troubleshooting

### No Audio Output
- Ensure pygame is installed: `pip install pygame`
- Check system audio is not muted
- Try exporting to file instead

### Export Errors
**WAV/MP3 Export Fails:**
- Install FluidSynth (system package)
- Install FFmpeg (system package)
- Check file permissions in output directory

### Training Issues
**Out of Memory:**
- Reduce batch size
- Use fewer layers in model
- Use LSTM instead of Transformer
- Enable gradient checkpointing

**Poor Quality Output:**
- Train longer (more epochs)
- Use more training data
- Ensure data quality (clean MIDI files)
- Try data augmentation

### GPU Not Detected
- Install CUDA-compatible PyTorch
- Update NVIDIA drivers
- Check CUDA installation: `torch.cuda.is_available()`

## Performance Tips

1. **Use GPU**: Significantly faster generation and training
2. **Adjust Creativity**: Lower values (0.7-0.9) = more coherent
3. **Shorter Sequences**: Start with 30-60 second compositions
4. **Pretrained Models**: Use provided models or train on large datasets
5. **Batch Processing**: Generate multiple compositions in sequence

## System Requirements

**Minimum:**
- CPU: Intel Core i5 or equivalent
- RAM: 8 GB
- Storage: 2 GB free space
- OS: Windows 11, Windows 10, Linux, macOS

**Recommended:**
- CPU: Intel Core i7 or equivalent
- GPU: NVIDIA GTX 1060 or better (CUDA support)
- RAM: 16 GB
- Storage: 5 GB free space
- OS: Windows 11

## License

This project is open source and available for educational and personal use.

## Credits

Built with:
- PyTorch (Deep Learning)
- midiutil & mido (MIDI Processing)
- pygame (Audio Playback)
- pydub (Audio Conversion)
- tkinter (GUI)

## Support

For issues, questions, or contributions:
1. Check this README for solutions
2. Review code comments and docstrings
3. Experiment with different parameters
4. Train custom models on your data

## Version History

**v1.0.0** - Initial Release
- LSTM and Transformer models
- 8 genres, 8 moods
- MIDI, WAV, MP3 export
- Custom training pipeline
- Full-featured GUI
- Windows 11 optimized

---

**Happy Composing! 🎵🎹🎶**
