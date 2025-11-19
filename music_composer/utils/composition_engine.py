"""
Main composition engine combining models and music theory
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from .music_theory import MusicTheory, GENRE_PARAMS, MOOD_PARAMS
from .midi_processor import MIDIProcessor


class CompositionEngine:
    """Main engine for music composition"""

    def __init__(self, model, model_type: str = 'transformer', device: str = 'cpu'):
        """
        Initialize composition engine

        Args:
            model: Trained model (LSTM or Transformer)
            model_type: 'lstm' or 'transformer'
            device: Device to run on
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.midi_processor = MIDIProcessor()
        self.music_theory = MusicTheory()

        # Genre and mood mappings
        self.genre_map = {
            'classical': 0, 'jazz': 1, 'pop': 2, 'rock': 3,
            'electronic': 4, 'blues': 5, 'ambient': 6, 'folk': 7
        }
        self.mood_map = {
            'happy': 0, 'sad': 1, 'energetic': 2, 'calm': 3,
            'mysterious': 4, 'romantic': 5, 'aggressive': 6, 'peaceful': 7
        }

    def generate_melody(
        self,
        genre: str = 'pop',
        mood: str = 'happy',
        length: int = 200,
        temperature: float = 1.0,
        seed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate melody

        Args:
            genre: Music genre
            mood: Mood/emotion
            length: Number of notes to generate
            temperature: Sampling temperature
            seed: Seed sequence

        Returns:
            Generated note sequence
        """
        # Get genre and mood parameters
        scale_name = self.music_theory.get_genre_scale(genre)
        mood_params = MOOD_PARAMS.get(mood, MOOD_PARAMS['happy'])

        # Override scale with mood if specified
        if 'scale' in mood_params:
            scale_name = mood_params['scale']

        # Create seed if not provided
        if seed is None:
            # Start with middle C and some scale notes
            root = 60
            scale_notes = self.music_theory.get_scale_notes(root, scale_name)
            seed = np.random.choice(scale_notes, size=10)

        seed_tensor = torch.tensor(seed, dtype=torch.long)

        # Generate using model
        if hasattr(self.model, 'generate_conditional'):
            # Use conditional generation
            genre_idx = self.genre_map.get(genre, 0)
            mood_idx = self.mood_map.get(mood, 0)

            generated = self.model.generate_conditional(
                seed_tensor,
                genre_idx,
                mood_idx,
                length,
                temperature,
                device=self.device
            )
        else:
            # Use basic generation
            generated = self.model.generate(
                seed_tensor,
                length,
                temperature,
                device=self.device
            )

        # Constrain to scale
        root = 60
        scale_notes = self.music_theory.get_scale_notes(root, scale_name)
        generated = self.music_theory.constrain_to_scale(generated, scale_notes)

        return generated

    def generate_harmony(
        self,
        melody: np.ndarray,
        genre: str = 'pop',
        root: int = 60
    ) -> np.ndarray:
        """Generate harmony for melody"""
        # Get chord progression
        progression = self.music_theory.get_progression(genre, root)

        # Add harmony
        _, harmony = self.music_theory.add_harmony(melody, progression)

        return harmony

    def compose(
        self,
        genre: str = 'pop',
        mood: str = 'happy',
        length: int = 200,
        temperature: float = 1.0,
        add_harmony: bool = True,
        add_drums: bool = False,
        tempo: int = 120
    ) -> Dict:
        """
        Full composition with melody, harmony, and drums

        Args:
            genre: Music genre
            mood: Mood
            length: Length in notes
            temperature: Sampling temperature
            add_harmony: Whether to add harmony
            add_drums: Whether to add drums
            tempo: Tempo in BPM

        Returns:
            Dictionary with composition data
        """
        # Generate melody
        melody = self.generate_melody(genre, mood, length, temperature)

        # Convert to notes with rhythm
        melody_notes = self.midi_processor.sequence_to_notes(melody, genre)

        tracks = [(melody_notes, 0)]  # Piano melody

        # Add harmony
        if add_harmony:
            harmony = self.generate_harmony(melody, genre)
            harmony_notes = self.midi_processor.sequence_to_notes(harmony, genre)
            tracks.append((harmony_notes, 48))  # String ensemble

        # Add drums
        if add_drums:
            # Calculate total beats
            total_duration = sum(d for _, d, _ in melody_notes)
            beats = int(total_duration / (60.0 / tempo))
            drum_notes = self.midi_processor.add_drums(beats, genre, tempo)
            tracks.append((drum_notes, 0))  # Drums on percussion channel

        return {
            'melody': melody,
            'melody_notes': melody_notes,
            'tracks': tracks,
            'tempo': tempo,
            'genre': genre,
            'mood': mood
        }

    def save_composition(
        self,
        composition: Dict,
        output_file: str
    ):
        """Save composition to MIDI file"""
        tracks = composition['tracks']
        tempo = composition['tempo']

        self.midi_processor.create_multi_track_midi(tracks, output_file, tempo)

    def load_and_train_data(self, midi_file: str) -> Tuple[np.ndarray, List[float], List[int]]:
        """Load MIDI file for training"""
        return self.midi_processor.midi_to_notes(midi_file)


class MoodBasedComposer:
    """Specialized composer for mood-based generation"""

    def __init__(self, engine: CompositionEngine):
        self.engine = engine

    def compose_for_mood(
        self,
        mood: str,
        duration: int = 60,  # seconds
        genre: str = 'ambient',
        tempo: Optional[int] = None
    ) -> Dict:
        """
        Compose music for specific mood

        Args:
            mood: Target mood
            duration: Duration in seconds
            genre: Music genre
            tempo: Tempo (auto-calculated if None)

        Returns:
            Composition dictionary
        """
        # Get mood parameters
        mood_params = MOOD_PARAMS.get(mood, MOOD_PARAMS['happy'])

        # Calculate tempo based on mood
        if tempo is None:
            genre_params = GENRE_PARAMS.get(genre, GENRE_PARAMS['pop'])
            base_tempo = sum(genre_params['tempo_range']) // 2
            tempo = int(base_tempo * mood_params['tempo_multiplier'])

        # Calculate number of notes needed
        avg_note_duration = 0.5  # seconds
        num_notes = int(duration / avg_note_duration)

        # Adjust temperature based on mood
        temperature_map = {
            'happy': 1.0,
            'sad': 0.8,
            'energetic': 1.2,
            'calm': 0.7,
            'mysterious': 1.1,
            'romantic': 0.9,
            'aggressive': 1.3,
            'peaceful': 0.7,
        }
        temperature = temperature_map.get(mood, 1.0)

        # Compose
        composition = self.engine.compose(
            genre=genre,
            mood=mood,
            length=num_notes,
            temperature=temperature,
            add_harmony=True,
            add_drums=(mood in ['energetic', 'aggressive', 'happy']),
            tempo=tempo
        )

        return composition

    def create_mood_journey(
        self,
        mood_sequence: List[str],
        duration_per_mood: int = 30,
        genre: str = 'ambient'
    ) -> List[Dict]:
        """
        Create a journey through different moods

        Args:
            mood_sequence: List of moods in order
            duration_per_mood: Duration per mood in seconds
            genre: Music genre

        Returns:
            List of compositions
        """
        compositions = []

        for mood in mood_sequence:
            comp = self.compose_for_mood(mood, duration_per_mood, genre)
            compositions.append(comp)

        return compositions
