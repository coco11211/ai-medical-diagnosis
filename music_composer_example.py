#!/usr/bin/env python3
"""
AI Music Composer - Example Usage Script
Demonstrates how to use the music composer programmatically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'music_composer'))

import torch
from music_composer.models.transformer_model import ConditionalTransformer
from music_composer.models.lstm_model import ConditionalLSTM
from music_composer.utils.composition_engine import CompositionEngine, MoodBasedComposer
from music_composer.export.audio_export import AudioExporter


def example_1_basic_generation():
    """Example 1: Basic music generation"""
    print("\n=== Example 1: Basic Music Generation ===\n")

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = ConditionalTransformer(
        vocab_size=128,
        d_model=256,
        nhead=8,
        num_layers=4
    ).to(device)

    # Create composition engine
    engine = CompositionEngine(model, 'transformer', device)

    # Generate melody
    print("Generating melody...")
    melody = engine.generate_melody(
        genre='pop',
        mood='happy',
        length=100,
        temperature=1.0
    )

    print(f"Generated {len(melody)} notes")

    # Create full composition
    print("Creating full composition...")
    composition = engine.compose(
        genre='pop',
        mood='happy',
        length=100,
        temperature=1.0,
        add_harmony=True,
        add_drums=False,
        tempo=120
    )

    # Save to MIDI
    output_file = "example_1_output.mid"
    engine.save_composition(composition, output_file)
    print(f"Saved to: {output_file}")


def example_2_mood_based():
    """Example 2: Mood-based composition"""
    print("\n=== Example 2: Mood-Based Composition ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model and composer
    model = ConditionalTransformer().to(device)
    engine = CompositionEngine(model, 'transformer', device)
    composer = MoodBasedComposer(engine)

    # Generate for different moods
    moods = ['happy', 'sad', 'energetic']

    for mood in moods:
        print(f"\nComposing {mood} music...")
        composition = composer.compose_for_mood(
            mood=mood,
            duration=30,
            genre='ambient',
            tempo=None  # Auto-calculated based on mood
        )

        output_file = f"example_2_{mood}.mid"
        engine.save_composition(composition, output_file)
        print(f"Saved {mood} composition to: {output_file}")


def example_3_multi_genre():
    """Example 3: Multi-genre compositions"""
    print("\n=== Example 3: Multi-Genre Compositions ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConditionalTransformer().to(device)
    engine = CompositionEngine(model, 'transformer', device)

    genres = ['classical', 'jazz', 'rock', 'electronic']

    for genre in genres:
        print(f"\nComposing {genre} music...")
        composition = engine.compose(
            genre=genre,
            mood='energetic',
            length=150,
            temperature=1.0,
            add_harmony=True,
            add_drums=True,
            tempo=120
        )

        output_file = f"example_3_{genre}.mid"
        engine.save_composition(composition, output_file)
        print(f"Saved {genre} composition to: {output_file}")


def example_4_export_audio():
    """Example 4: Export to audio formats"""
    print("\n=== Example 4: Export to Audio Formats ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate composition
    model = ConditionalTransformer().to(device)
    engine = CompositionEngine(model, 'transformer', device)

    print("Generating composition...")
    composition = engine.compose(
        genre='pop',
        mood='romantic',
        length=120,
        temperature=0.9,
        add_harmony=True,
        add_drums=False,
        tempo=90
    )

    # Save MIDI
    midi_file = "example_4_output.mid"
    engine.save_composition(composition, midi_file)
    print(f"Saved MIDI: {midi_file}")

    # Export to WAV
    exporter = AudioExporter()
    try:
        wav_file = "example_4_output.wav"
        print(f"Converting to WAV...")
        exporter.midi_to_wav(midi_file, wav_file)
        print(f"Saved WAV: {wav_file}")

        # Export to MP3
        mp3_file = "example_4_output.mp3"
        print(f"Converting to MP3...")
        exporter.wav_to_mp3(wav_file, mp3_file)
        print(f"Saved MP3: {mp3_file}")

    except Exception as e:
        print(f"Audio export requires additional dependencies: {e}")
        print("Install FluidSynth and FFmpeg for audio export")


def example_5_lstm_model():
    """Example 5: Using LSTM model"""
    print("\n=== Example 5: LSTM Model Generation ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create LSTM model
    model = ConditionalLSTM(
        vocab_size=128,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=3
    ).to(device)

    engine = CompositionEngine(model, 'lstm', device)

    print("Generating with LSTM model...")
    composition = engine.compose(
        genre='blues',
        mood='calm',
        length=100,
        temperature=1.0,
        add_harmony=True,
        add_drums=True,
        tempo=80
    )

    output_file = "example_5_lstm_output.mid"
    engine.save_composition(composition, output_file)
    print(f"Saved LSTM composition to: {output_file}")


def example_6_mood_journey():
    """Example 6: Create a mood journey"""
    print("\n=== Example 6: Mood Journey ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConditionalTransformer().to(device)
    engine = CompositionEngine(model, 'transformer', device)
    composer = MoodBasedComposer(engine)

    # Create a journey through moods
    mood_sequence = ['calm', 'mysterious', 'energetic', 'peaceful']

    print("Creating mood journey...")
    compositions = composer.create_mood_journey(
        mood_sequence=mood_sequence,
        duration_per_mood=20,
        genre='ambient'
    )

    # Save each part
    for i, (mood, comp) in enumerate(zip(mood_sequence, compositions)):
        output_file = f"example_6_journey_part{i+1}_{mood}.mid"
        engine.save_composition(comp, output_file)
        print(f"Saved part {i+1} ({mood}): {output_file}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("AI Music Composer - Example Usage")
    print("=" * 60)

    examples = [
        ("Basic Generation", example_1_basic_generation),
        ("Mood-Based Composition", example_2_mood_based),
        ("Multi-Genre Compositions", example_3_multi_genre),
        ("Export to Audio Formats", example_4_export_audio),
        ("LSTM Model", example_5_lstm_model),
        ("Mood Journey", example_6_mood_journey),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nRunning Example 1 (Basic Generation)...")
    print("To run other examples, modify the main() function\n")

    # Run example 1 by default
    example_1_basic_generation()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTo run the GUI application, use:")
    print("  python music_composer_app.py")
    print("\nGenerated files can be opened with any MIDI player or DAW")


if __name__ == "__main__":
    main()
