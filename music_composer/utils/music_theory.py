"""
Music theory utilities for harmony and melody generation
"""

import numpy as np
from typing import List, Dict, Tuple


# Musical scales (intervals from root)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'chromatic': list(range(12))
}

# Chord types (intervals from root)
CHORDS = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented': [0, 4, 8],
    'major7': [0, 4, 7, 11],
    'minor7': [0, 3, 7, 10],
    'dominant7': [0, 4, 7, 10],
    'diminished7': [0, 3, 6, 9],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'major6': [0, 4, 7, 9],
    'minor6': [0, 3, 7, 9],
}

# Common chord progressions by genre
PROGRESSIONS = {
    'pop': [
        ['I', 'V', 'vi', 'IV'],  # I-V-vi-IV
        ['I', 'IV', 'V', 'I'],    # I-IV-V-I
        ['vi', 'IV', 'I', 'V'],   # vi-IV-I-V
        ['I', 'vi', 'IV', 'V'],   # I-vi-IV-V
    ],
    'jazz': [
        ['IIm7', 'V7', 'Imaj7'],
        ['Imaj7', 'VIm7', 'IIm7', 'V7'],
        ['IIIm7', 'VI7', 'IIm7', 'V7'],
    ],
    'blues': [
        ['I7', 'I7', 'I7', 'I7', 'IV7', 'IV7', 'I7', 'I7', 'V7', 'IV7', 'I7', 'V7'],
    ],
    'rock': [
        ['I', 'bVII', 'IV', 'I'],
        ['I', 'IV', 'I', 'V'],
        ['I', 'bIII', 'bVII', 'IV'],
    ],
    'classical': [
        ['I', 'IV', 'V', 'I'],
        ['I', 'V', 'I'],
        ['I', 'IV', 'I', 'V', 'I'],
    ]
}

# Genre characteristics
GENRE_PARAMS = {
    'classical': {
        'scale': 'major',
        'tempo_range': (60, 120),
        'complexity': 'high',
        'dynamics_range': (30, 100),
    },
    'jazz': {
        'scale': 'melodic_minor',
        'tempo_range': (100, 180),
        'complexity': 'very_high',
        'dynamics_range': (40, 90),
    },
    'pop': {
        'scale': 'major',
        'tempo_range': (100, 130),
        'complexity': 'medium',
        'dynamics_range': (60, 95),
    },
    'rock': {
        'scale': 'pentatonic_minor',
        'tempo_range': (110, 160),
        'complexity': 'medium',
        'dynamics_range': (70, 100),
    },
    'electronic': {
        'scale': 'minor',
        'tempo_range': (120, 150),
        'complexity': 'medium',
        'dynamics_range': (70, 100),
    },
    'blues': {
        'scale': 'blues',
        'tempo_range': (60, 120),
        'complexity': 'medium',
        'dynamics_range': (50, 85),
    },
    'ambient': {
        'scale': 'lydian',
        'tempo_range': (60, 90),
        'complexity': 'low',
        'dynamics_range': (20, 60),
    },
    'folk': {
        'scale': 'major',
        'tempo_range': (80, 120),
        'complexity': 'low',
        'dynamics_range': (40, 75),
    },
}

# Mood to musical parameters
MOOD_PARAMS = {
    'happy': {
        'scale': 'major',
        'tempo_multiplier': 1.2,
        'brightness': 0.8,
        'chord_type': 'major',
    },
    'sad': {
        'scale': 'minor',
        'tempo_multiplier': 0.7,
        'brightness': 0.3,
        'chord_type': 'minor',
    },
    'energetic': {
        'scale': 'mixolydian',
        'tempo_multiplier': 1.4,
        'brightness': 0.9,
        'chord_type': 'major',
    },
    'calm': {
        'scale': 'lydian',
        'tempo_multiplier': 0.8,
        'brightness': 0.5,
        'chord_type': 'major',
    },
    'mysterious': {
        'scale': 'phrygian',
        'tempo_multiplier': 0.9,
        'brightness': 0.4,
        'chord_type': 'diminished',
    },
    'romantic': {
        'scale': 'major',
        'tempo_multiplier': 0.9,
        'brightness': 0.7,
        'chord_type': 'major7',
    },
    'aggressive': {
        'scale': 'phrygian',
        'tempo_multiplier': 1.5,
        'brightness': 0.6,
        'chord_type': 'diminished',
    },
    'peaceful': {
        'scale': 'major',
        'tempo_multiplier': 0.7,
        'brightness': 0.6,
        'chord_type': 'major',
    },
}


class MusicTheory:
    """Music theory helper class"""

    @staticmethod
    def get_scale_notes(root: int, scale_name: str) -> List[int]:
        """Get all notes in a scale across octaves"""
        intervals = SCALES.get(scale_name, SCALES['major'])
        notes = []
        for octave in range(11):  # MIDI range
            base = octave * 12 + root
            for interval in intervals:
                note = base + interval
                if 0 <= note < 128:
                    notes.append(note)
        return notes

    @staticmethod
    def get_chord(root: int, chord_type: str) -> List[int]:
        """Get chord notes"""
        intervals = CHORDS.get(chord_type, CHORDS['major'])
        return [root + i for i in intervals if root + i < 128]

    @staticmethod
    def get_progression(genre: str, root: int = 60) -> List[List[int]]:
        """Get chord progression for genre"""
        if genre not in PROGRESSIONS:
            genre = 'pop'

        progressions = PROGRESSIONS[genre]
        progression = progressions[np.random.randint(len(progressions))]

        # Convert roman numerals to actual chords
        degree_map = {
            'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11,
            'i': 0, 'ii': 2, 'iii': 4, 'iv': 5, 'v': 7, 'vi': 9, 'vii': 11,
            'bII': 1, 'bIII': 3, 'bV': 6, 'bVI': 8, 'bVII': 10,
        }

        chords = []
        for numeral in progression:
            # Parse chord symbol
            base_numeral = numeral.rstrip('m7maj6sus24')
            degree = degree_map.get(base_numeral, 0)
            chord_root = root + degree

            # Determine chord type
            if 'maj7' in numeral:
                chord_type = 'major7'
            elif 'm7' in numeral:
                chord_type = 'minor7'
            elif '7' in numeral:
                chord_type = 'dominant7'
            elif 'm' in numeral:
                chord_type = 'minor'
            else:
                chord_type = 'major'

            chord = MusicTheory.get_chord(chord_root, chord_type)
            chords.append(chord)

        return chords

    @staticmethod
    def constrain_to_scale(notes: np.ndarray, scale_notes: List[int]) -> np.ndarray:
        """Constrain notes to a scale"""
        constrained = np.zeros_like(notes)
        scale_set = set(n % 12 for n in scale_notes)

        for i, note in enumerate(notes):
            if note % 12 in scale_set:
                constrained[i] = note
            else:
                # Find nearest scale note
                octave = (note // 12) * 12
                pitch_class = note % 12
                distances = [abs(pitch_class - s % 12) for s in scale_notes]
                nearest = scale_notes[np.argmin(distances)]
                constrained[i] = octave + (nearest % 12)

        return constrained

    @staticmethod
    def add_harmony(melody: np.ndarray, chord_progression: List[List[int]],
                   notes_per_chord: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Add harmony to melody"""
        num_chords = len(chord_progression)
        melody_len = len(melody)
        chord_duration = melody_len // num_chords

        harmony = []
        for i, chord in enumerate(chord_progression):
            for _ in range(chord_duration):
                # Pick random note from chord (typically root or third)
                harmony_note = chord[np.random.choice([0, 1])]
                harmony.append(harmony_note)

        # Pad if necessary
        while len(harmony) < melody_len:
            harmony.append(harmony[-1])

        return melody, np.array(harmony[:melody_len])

    @staticmethod
    def get_genre_scale(genre: str) -> str:
        """Get appropriate scale for genre"""
        return GENRE_PARAMS.get(genre, GENRE_PARAMS['pop'])['scale']

    @staticmethod
    def get_mood_scale(mood: str) -> str:
        """Get appropriate scale for mood"""
        return MOOD_PARAMS.get(mood, MOOD_PARAMS['happy'])['scale']

    @staticmethod
    def apply_rhythm(notes: np.ndarray, genre: str) -> List[Tuple[int, float, float]]:
        """
        Apply rhythm to notes
        Returns list of (note, duration, velocity)
        """
        tempo_range = GENRE_PARAMS.get(genre, GENRE_PARAMS['pop'])['tempo_range']
        tempo = np.random.randint(*tempo_range)

        # Convert tempo to note duration (in seconds)
        quarter_note = 60.0 / tempo

        rhythm_patterns = {
            'classical': [1.0, 1.0, 0.5, 0.5, 1.0],
            'jazz': [0.75, 0.25, 1.0, 0.5, 0.5],
            'pop': [1.0, 1.0, 0.5, 0.5],
            'rock': [1.0, 0.5, 0.5, 1.0],
            'electronic': [0.25, 0.25, 0.25, 0.25],
            'blues': [1.0, 1.0, 0.5, 0.5],
        }

        pattern = rhythm_patterns.get(genre, rhythm_patterns['pop'])
        dynamics_range = GENRE_PARAMS.get(genre, GENRE_PARAMS['pop'])['dynamics_range']

        result = []
        pattern_idx = 0

        for note in notes:
            duration = pattern[pattern_idx % len(pattern)] * quarter_note
            velocity = np.random.randint(*dynamics_range)
            result.append((int(note), float(duration), int(velocity)))
            pattern_idx += 1

        return result
