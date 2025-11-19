"""
Audio Utilities
Helper functions for audio processing, format conversion, and analysis
"""

import numpy as np
import wave
from pathlib import Path
from typing import Tuple, Optional
import json


class AudioMetrics:
    """Calculate audio quality metrics"""

    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR)

        Args:
            signal: Clean signal
            noise: Noise signal

        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)

        if noise_power < 1e-10:
            return np.inf

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) level

        Args:
            audio: Audio signal

        Returns:
            RMS level
        """
        return np.sqrt(np.mean(audio**2))

    @staticmethod
    def calculate_peak(audio: np.ndarray) -> float:
        """
        Calculate peak level

        Args:
            audio: Audio signal

        Returns:
            Peak level
        """
        return np.max(np.abs(audio))

    @staticmethod
    def calculate_crest_factor(audio: np.ndarray) -> float:
        """
        Calculate crest factor (peak-to-RMS ratio)

        Args:
            audio: Audio signal

        Returns:
            Crest factor in dB
        """
        rms = AudioMetrics.calculate_rms(audio)
        peak = AudioMetrics.calculate_peak(audio)

        if rms < 1e-10:
            return 0.0

        crest_factor = 20 * np.log10(peak / rms)
        return crest_factor

    @staticmethod
    def calculate_thd(audio: np.ndarray, sample_rate: int, fundamental_freq: float) -> float:
        """
        Calculate Total Harmonic Distortion (THD)

        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz
            fundamental_freq: Fundamental frequency in Hz

        Returns:
            THD as percentage
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

        # Find fundamental frequency bin
        fundamental_bin = np.argmin(np.abs(freqs - fundamental_freq))
        fundamental_magnitude = np.abs(fft[fundamental_bin])

        # Calculate harmonic magnitudes (up to 5th harmonic)
        harmonic_power = 0.0
        for n in range(2, 6):
            harmonic_freq = n * fundamental_freq
            if harmonic_freq > sample_rate / 2:
                break

            harmonic_bin = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_magnitude = np.abs(fft[harmonic_bin])
            harmonic_power += harmonic_magnitude**2

        # Calculate THD
        if fundamental_magnitude < 1e-10:
            return 0.0

        thd = 100 * np.sqrt(harmonic_power) / fundamental_magnitude
        return thd

    @staticmethod
    def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict:
        """
        Comprehensive audio analysis

        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with analysis results
        """
        return {
            'rms_db': 20 * np.log10(max(AudioMetrics.calculate_rms(audio), 1e-10)),
            'peak_db': 20 * np.log10(max(AudioMetrics.calculate_peak(audio), 1e-10)),
            'crest_factor_db': AudioMetrics.calculate_crest_factor(audio),
            'duration_s': len(audio) / sample_rate,
            'sample_count': len(audio),
            'sample_rate': sample_rate,
            'dynamic_range_db': (
                20 * np.log10(max(AudioMetrics.calculate_peak(audio), 1e-10)) -
                20 * np.log10(max(AudioMetrics.calculate_rms(audio), 1e-10))
            )
        }


class AudioFileHandler:
    """Handle audio file operations"""

    @staticmethod
    def read_wav(file_path: str) -> Tuple[np.ndarray, int, int]:
        """
        Read WAV file

        Args:
            file_path: Path to WAV file

        Returns:
            Tuple of (audio_data, sample_rate, channels)
        """
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

            # Convert to numpy array
            if sample_width == 1:  # 8-bit
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = (audio_data.astype(np.float32) - 128) / 128
            elif sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32)
                audio_data = audio_data.astype(np.float32) / 2147483648
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            return audio_data, sample_rate, channels

    @staticmethod
    def write_wav(file_path: str, audio_data: np.ndarray, sample_rate: int,
                  channels: int = 1, bit_depth: int = 16) -> None:
        """
        Write WAV file

        Args:
            file_path: Path to output WAV file
            audio_data: Audio data as float32 (-1.0 to 1.0)
            sample_rate: Sample rate in Hz
            channels: Number of channels
            bit_depth: Bit depth (8, 16, or 32)
        """
        # Create output directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Clip audio data
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Convert to appropriate integer type
        if bit_depth == 8:
            audio_int = ((audio_data + 1.0) * 128).astype(np.uint8)
            sample_width = 1
        elif bit_depth == 16:
            audio_int = (audio_data * 32767).astype(np.int16)
            sample_width = 2
        elif bit_depth == 32:
            audio_int = (audio_data * 2147483647).astype(np.int32)
            sample_width = 4
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        # Write WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int.tobytes())

    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Get audio file information

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with file information
        """
        with wave.open(file_path, 'rb') as wf:
            return {
                'sample_rate': wf.getframerate(),
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'frame_count': wf.getnframes(),
                'duration_s': wf.getnframes() / wf.getframerate(),
                'bit_depth': wf.getsampwidth() * 8
            }


class AudioConverter:
    """Audio format conversion utilities"""

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to different sample rate

        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        from scipy import signal

        if orig_sr == target_sr:
            return audio

        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        num_samples = int(len(audio) * ratio)

        # Use scipy's resample
        resampled = signal.resample(audio, num_samples)

        return resampled

    @staticmethod
    def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
        """
        Convert mono to stereo

        Args:
            audio: Mono audio

        Returns:
            Stereo audio
        """
        return np.stack([audio, audio], axis=1)

    @staticmethod
    def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo to mono

        Args:
            audio: Stereo audio

        Returns:
            Mono audio
        """
        if len(audio.shape) == 1:
            return audio

        return np.mean(audio, axis=1)

    @staticmethod
    def normalize(audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target level

        Args:
            audio: Input audio
            target_level: Target level in dB

        Returns:
            Normalized audio
        """
        # Calculate current peak level
        peak = np.max(np.abs(audio))

        if peak < 1e-10:
            return audio

        # Calculate target gain
        target_amplitude = 10 ** (target_level / 20)
        gain = target_amplitude / peak

        # Apply gain
        normalized = audio * gain

        return np.clip(normalized, -1.0, 1.0)


def save_processing_report(input_file: str, output_file: str,
                           processing_stats: dict, output_path: str = "processing_report.json"):
    """
    Save audio processing report

    Args:
        input_file: Input file path
        output_file: Output file path
        processing_stats: Processing statistics
        output_path: Report output path
    """
    report = {
        'input_file': input_file,
        'output_file': output_file,
        'processing_stats': processing_stats,
        'timestamp': str(np.datetime64('now'))
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Processing report saved to: {output_path}")
