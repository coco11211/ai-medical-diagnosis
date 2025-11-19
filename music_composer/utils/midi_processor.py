"""
MIDI processing and generation utilities
"""

import numpy as np
from typing import List, Tuple, Optional
import os


class MIDIProcessor:
    """Process and generate MIDI files"""

    def __init__(self):
        self.ticks_per_beat = 480
        self.tempo = 500000  # microseconds per quarter note (120 BPM)

    def notes_to_midi(
        self,
        notes: List[Tuple[int, float, int]],
        output_file: str,
        instrument: int = 0,
        tempo: int = 120
    ):
        """
        Convert note sequence to MIDI file

        Args:
            notes: List of (note, duration, velocity) tuples
            output_file: Output MIDI file path
            instrument: MIDI instrument number (0-127)
            tempo: Tempo in BPM
        """
        try:
            from midiutil import MIDIFile
        except ImportError:
            raise ImportError("midiutil not installed. Install with: pip install midiutil")

        # Create MIDI file with 1 track
        midi = MIDIFile(1)

        track = 0
        channel = 0
        time = 0

        # Set tempo
        midi.addTempo(track, time, tempo)

        # Set instrument
        midi.addProgramChange(track, channel, time, instrument)

        # Add notes
        current_time = 0
        for note, duration, velocity in notes:
            if 0 <= note < 128:  # Valid MIDI note range
                midi.addNote(track, channel, note, current_time, duration, velocity)
                current_time += duration

        # Write to file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'wb') as f:
            midi.writeFile(f)

    def midi_to_notes(self, midi_file: str) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Extract notes from MIDI file

        Args:
            midi_file: Path to MIDI file

        Returns:
            Tuple of (notes, durations, velocities)
        """
        try:
            import mido
        except ImportError:
            raise ImportError("mido not installed. Install with: pip install mido")

        midi = mido.MidiFile(midi_file)
        notes = []
        durations = []
        velocities = []

        current_notes = {}  # track -> {note: (time, velocity)}
        current_time = 0

        for track in midi.tracks:
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    current_notes[msg.note] = (current_time, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in current_notes:
                        start_time, velocity = current_notes[msg.note]
                        duration = current_time - start_time

                        notes.append(msg.note)
                        durations.append(duration)
                        velocities.append(velocity)

                        del current_notes[msg.note]

        return np.array(notes), durations, velocities

    def create_multi_track_midi(
        self,
        tracks: List[Tuple[List[Tuple[int, float, int]], int]],
        output_file: str,
        tempo: int = 120
    ):
        """
        Create multi-track MIDI file

        Args:
            tracks: List of (notes, instrument) tuples
            output_file: Output file path
            tempo: Tempo in BPM
        """
        try:
            from midiutil import MIDIFile
        except ImportError:
            raise ImportError("midiutil not installed. Install with: pip install midiutil")

        num_tracks = len(tracks)
        midi = MIDIFile(num_tracks)

        for track_idx, (notes, instrument) in enumerate(tracks):
            channel = track_idx % 16  # MIDI has 16 channels

            # Set tempo (only needed once)
            if track_idx == 0:
                midi.addTempo(track_idx, 0, tempo)

            # Set instrument
            midi.addProgramChange(track_idx, channel, 0, instrument)

            # Add notes
            current_time = 0
            for note, duration, velocity in notes:
                if 0 <= note < 128:
                    midi.addNote(track_idx, channel, note, current_time, duration, velocity)
                    current_time += duration

        # Write file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'wb') as f:
            midi.writeFile(f)

    def sequence_to_notes(
        self,
        sequence: np.ndarray,
        genre: str = 'pop',
        add_rhythm: bool = True
    ) -> List[Tuple[int, float, int]]:
        """Convert note sequence to (note, duration, velocity) format"""
        from .music_theory import MusicTheory

        if add_rhythm:
            return MusicTheory.apply_rhythm(sequence, genre)
        else:
            # Default rhythm
            notes = []
            for note in sequence:
                notes.append((int(note), 0.5, 80))
            return notes

    def add_drums(
        self,
        length: int,
        genre: str = 'rock',
        tempo: int = 120
    ) -> List[Tuple[int, float, int]]:
        """
        Generate drum pattern

        Args:
            length: Number of beats
            genre: Music genre
            tempo: Tempo in BPM

        Returns:
            List of drum notes
        """
        # MIDI drum notes (General MIDI percussion)
        KICK = 36
        SNARE = 38
        CLOSED_HAT = 42
        OPEN_HAT = 46
        CRASH = 49

        drums = []
        beat_duration = 60.0 / tempo

        drum_patterns = {
            'rock': [
                (KICK, 1.0, 100),
                (CLOSED_HAT, 0.5, 80),
                (SNARE, 1.0, 90),
                (CLOSED_HAT, 0.5, 80),
            ],
            'pop': [
                (KICK, 1.0, 90),
                (CLOSED_HAT, 0.5, 70),
                (SNARE, 1.0, 85),
                (CLOSED_HAT, 0.5, 70),
            ],
            'jazz': [
                (KICK, 1.0, 70),
                (CLOSED_HAT, 0.333, 60),
                (CLOSED_HAT, 0.333, 60),
                (CLOSED_HAT, 0.333, 60),
            ],
            'electronic': [
                (KICK, 0.25, 100),
                (CLOSED_HAT, 0.25, 90),
                (SNARE, 0.25, 95),
                (CLOSED_HAT, 0.25, 90),
            ],
        }

        pattern = drum_patterns.get(genre, drum_patterns['pop'])

        for i in range(length):
            for note, duration, velocity in pattern:
                drums.append((note, duration * beat_duration, velocity))

        return drums


class MIDIAugmenter:
    """Augment MIDI data for training"""

    @staticmethod
    def transpose(notes: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose notes by semitones"""
        transposed = notes + semitones
        return np.clip(transposed, 0, 127)

    @staticmethod
    def time_stretch(durations: List[float], factor: float) -> List[float]:
        """Stretch time by factor"""
        return [d * factor for d in durations]

    @staticmethod
    def velocity_shift(velocities: List[int], shift: int) -> List[int]:
        """Shift velocities"""
        shifted = [v + shift for v in velocities]
        return [max(1, min(127, v)) for v in shifted]

    @staticmethod
    def random_augmentation(
        notes: np.ndarray,
        durations: List[float],
        velocities: List[int]
    ) -> Tuple[np.ndarray, List[float], List[int]]:
        """Apply random augmentation"""
        # Random transpose (-3 to +3 semitones)
        transpose_amount = np.random.randint(-3, 4)
        notes = MIDIAugmenter.transpose(notes, transpose_amount)

        # Random time stretch (0.9 to 1.1)
        stretch_factor = np.random.uniform(0.9, 1.1)
        durations = MIDIAugmenter.time_stretch(durations, stretch_factor)

        # Random velocity shift (-10 to +10)
        velocity_shift = np.random.randint(-10, 11)
        velocities = MIDIAugmenter.velocity_shift(velocities, velocity_shift)

        return notes, durations, velocities
