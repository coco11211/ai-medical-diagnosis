"""
Audio export utilities for WAV and MP3
"""

import os
import numpy as np
from typing import List, Tuple, Optional


class AudioExporter:
    """Export MIDI to audio formats"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def midi_to_wav(
        self,
        midi_file: str,
        wav_file: str,
        soundfont: Optional[str] = None
    ):
        """
        Convert MIDI to WAV using FluidSynth

        Args:
            midi_file: Input MIDI file
            wav_file: Output WAV file
            soundfont: Path to soundfont file (.sf2)
        """
        try:
            import subprocess

            # Use FluidSynth command line
            if soundfont is None:
                # Try to find default soundfont
                soundfont = self._find_soundfont()

            if soundfont and os.path.exists(soundfont):
                cmd = [
                    'fluidsynth',
                    '-ni',
                    soundfont,
                    midi_file,
                    '-F', wav_file,
                    '-r', str(self.sample_rate)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # Fallback: synthesize with simple method
                self._synthesize_midi_to_wav(midi_file, wav_file)

        except (FileNotFoundError, subprocess.CalledProcessError):
            # FluidSynth not available, use fallback
            self._synthesize_midi_to_wav(midi_file, wav_file)

    def _synthesize_midi_to_wav(self, midi_file: str, wav_file: str):
        """Synthesize MIDI to WAV using basic synthesis"""
        try:
            import scipy.io.wavfile as wavfile
            import mido
        except ImportError:
            raise ImportError("scipy and mido required. Install with: pip install scipy mido")

        # Load MIDI
        midi = mido.MidiFile(midi_file)

        # Extract notes with timing
        notes = []
        tempo = 500000  # default tempo
        current_time = 0

        for track in midi.tracks:
            for msg in track:
                current_time += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)

                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'start': current_time,
                        'channel': msg.channel
                    })
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Find matching note_on
                    for n in reversed(notes):
                        if n['note'] == msg.note and 'end' not in n:
                            n['end'] = current_time
                            break

        # Synthesize audio
        if notes:
            max_time = max(n.get('end', n['start']) for n in notes)
            audio = self._synthesize_notes(notes, max_time)

            # Normalize
            audio = audio / np.max(np.abs(audio))
            audio = (audio * 32767).astype(np.int16)

            # Save WAV
            wavfile.write(wav_file, self.sample_rate, audio)

    def _synthesize_notes(self, notes: List[dict], duration: float) -> np.ndarray:
        """Synthesize notes using additive synthesis"""
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)

        for note_info in notes:
            if 'end' not in note_info:
                continue

            note = note_info['note']
            velocity = note_info['velocity']
            start = note_info['start']
            end = note_info['end']
            channel = note_info.get('channel', 0)

            # Calculate frequency
            freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

            # Time array
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            duration_samples = end_sample - start_sample

            if duration_samples <= 0:
                continue

            t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)

            # Generate waveform based on channel
            if channel == 9:  # Drum channel
                # Noise-based synthesis for drums
                wave = np.random.randn(duration_samples) * 0.5
            else:
                # Harmonic synthesis for melodic instruments
                wave = np.zeros(duration_samples)

                # Add harmonics
                for harmonic in range(1, 5):
                    amplitude = (velocity / 127.0) / harmonic
                    wave += amplitude * np.sin(2 * np.pi * freq * harmonic * t)

            # Apply envelope (ADSR)
            envelope = self._create_envelope(duration_samples)
            wave *= envelope

            # Mix into audio
            if start_sample + len(wave) <= len(audio):
                audio[start_sample:start_sample + len(wave)] += wave

        return audio

    def _create_envelope(self, num_samples: int) -> np.ndarray:
        """Create ADSR envelope"""
        # Simple envelope
        attack_samples = min(int(0.01 * self.sample_rate), num_samples // 4)
        decay_samples = min(int(0.05 * self.sample_rate), num_samples // 4)
        release_samples = min(int(0.1 * self.sample_rate), num_samples // 4)

        envelope = np.ones(num_samples)

        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        if decay_samples > 0:
            decay_start = attack_samples
            decay_end = attack_samples + decay_samples
            envelope[decay_start:decay_end] = np.linspace(1, 0.7, decay_samples)

        # Release
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(0.7, 0, release_samples)

        return envelope

    def wav_to_mp3(self, wav_file: str, mp3_file: str, bitrate: str = '192k'):
        """
        Convert WAV to MP3 using pydub

        Args:
            wav_file: Input WAV file
            mp3_file: Output MP3 file
            bitrate: MP3 bitrate
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub required. Install with: pip install pydub")

        try:
            # Load WAV
            audio = AudioSegment.from_wav(wav_file)

            # Export as MP3
            audio.export(mp3_file, format='mp3', bitrate=bitrate)

        except Exception as e:
            # Fallback to ffmpeg command line
            import subprocess
            cmd = ['ffmpeg', '-i', wav_file, '-b:a', bitrate, mp3_file, '-y']
            subprocess.run(cmd, check=True, capture_output=True)

    def midi_to_mp3(
        self,
        midi_file: str,
        mp3_file: str,
        soundfont: Optional[str] = None,
        bitrate: str = '192k'
    ):
        """
        Convert MIDI directly to MP3

        Args:
            midi_file: Input MIDI file
            mp3_file: Output MP3 file
            soundfont: Soundfont file
            bitrate: MP3 bitrate
        """
        # Create temporary WAV file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav = tmp.name

        try:
            # Convert to WAV
            self.midi_to_wav(midi_file, tmp_wav, soundfont)

            # Convert to MP3
            self.wav_to_mp3(tmp_wav, mp3_file, bitrate)

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    def _find_soundfont(self) -> Optional[str]:
        """Try to find a system soundfont"""
        common_paths = [
            '/usr/share/soundfonts/default.sf2',
            '/usr/share/sounds/sf2/default.sf2',
            '/usr/share/soundfonts/FluidR3_GM.sf2',
            'C:\\soundfonts\\default.sf2',
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None


class AudioPlayer:
    """Real-time audio playback"""

    def __init__(self):
        self.is_playing = False

    def play_midi(self, midi_file: str, soundfont: Optional[str] = None):
        """
        Play MIDI file in real-time

        Args:
            midi_file: MIDI file to play
            soundfont: Soundfont file
        """
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(midi_file)
            pygame.mixer.music.play()

            self.is_playing = True

        except ImportError:
            # Fallback to system player
            self._play_with_system(midi_file)

    def play_wav(self, wav_file: str):
        """Play WAV file"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(wav_file)
            pygame.mixer.music.play()

            self.is_playing = True

        except ImportError:
            self._play_with_system(wav_file)

    def stop(self):
        """Stop playback"""
        try:
            import pygame
            pygame.mixer.music.stop()
        except:
            pass

        self.is_playing = False

    def _play_with_system(self, audio_file: str):
        """Play using system default player"""
        import subprocess
        import platform

        system = platform.system()

        try:
            if system == 'Windows':
                os.startfile(audio_file)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', audio_file])
            else:  # Linux
                subprocess.run(['xdg-open', audio_file])
        except Exception as e:
            print(f"Could not play audio: {e}")

    def is_busy(self) -> bool:
        """Check if audio is playing"""
        try:
            import pygame
            return pygame.mixer.music.get_busy()
        except:
            return False
