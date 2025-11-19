"""
Noise Cancellation Module
Implements multiple noise reduction algorithms for real-time audio processing
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft
from typing import Optional, Tuple
import noisereduce as nr
from .audio_config import AudioConfig


class NoiseCanceller:
    """
    Multi-algorithm noise cancellation system
    Supports spectral subtraction, Wiener filtering, and noisereduce
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize noise canceller

        Args:
            config: AudioConfig object with processing settings
        """
        self.config = config
        self.noise_profile = None
        self.prev_noise_spectrum = None
        self.smoothed_spectrum = None

        # Wiener filter state
        self.noise_psd = None
        self.signal_psd = None

        # Spectral subtraction parameters
        self.alpha = config.noise_reduction_strength
        self.beta = 0.02  # Over-subtraction factor

    def set_noise_profile(self, noise_audio: np.ndarray) -> None:
        """
        Set noise profile for adaptive noise cancellation

        Args:
            noise_audio: Audio containing only noise
        """
        self.noise_profile = noise_audio

        # Compute noise spectrum
        noise_stft = self._compute_stft(noise_audio)
        self.prev_noise_spectrum = np.abs(noise_stft)
        self.smoothed_spectrum = self.prev_noise_spectrum.copy()

        # Initialize Wiener filter noise PSD
        self.noise_psd = np.mean(self.prev_noise_spectrum**2, axis=1, keepdims=True)

        print("Noise profile set successfully")

    def spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction noise reduction

        Args:
            audio: Input audio chunk

        Returns:
            Cleaned audio
        """
        if self.noise_profile is None:
            print("Warning: No noise profile set, returning original audio")
            return audio

        # Compute STFT
        audio_stft = self._compute_stft(audio)
        magnitude = np.abs(audio_stft)
        phase = np.angle(audio_stft)

        # Estimate noise from profile
        noise_magnitude = self.prev_noise_spectrum

        # Ensure shapes match
        if noise_magnitude.shape != magnitude.shape:
            # Use mean across time for noise
            noise_magnitude = np.mean(self.prev_noise_spectrum, axis=1, keepdims=True)
            noise_magnitude = np.tile(noise_magnitude, (1, magnitude.shape[1]))

        # Spectral subtraction with over-subtraction
        subtracted = magnitude - self.alpha * noise_magnitude - self.beta

        # Half-wave rectification (ensure non-negative)
        cleaned_magnitude = np.maximum(subtracted, 0.0)

        # Apply spectral floor (avoid complete silence)
        spectral_floor = 0.002 * magnitude
        cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)

        # Reconstruct with original phase
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)

        # Inverse STFT
        cleaned_audio = self._inverse_stft(cleaned_stft, len(audio))

        return cleaned_audio

    def wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filter for noise reduction

        Args:
            audio: Input audio chunk

        Returns:
            Cleaned audio
        """
        if self.noise_psd is None:
            print("Warning: No noise profile set, returning original audio")
            return audio

        # Compute STFT
        audio_stft = self._compute_stft(audio)
        magnitude = np.abs(audio_stft)
        phase = np.angle(audio_stft)

        # Estimate signal PSD
        signal_psd_estimate = magnitude**2

        # Smooth PSD estimates
        if self.signal_psd is None:
            self.signal_psd = signal_psd_estimate
        else:
            self.signal_psd = (
                self.config.wiener_alpha * self.signal_psd +
                (1 - self.config.wiener_alpha) * signal_psd_estimate
            )

        # Compute Wiener gain
        # G = S / (S + N) where S is signal PSD, N is noise PSD
        noise_psd_broadcast = np.tile(self.noise_psd, (1, self.signal_psd.shape[1]))
        wiener_gain = self.signal_psd / (self.signal_psd + noise_psd_broadcast + 1e-10)

        # Apply gain floor
        wiener_gain = np.maximum(wiener_gain, self.config.wiener_beta)

        # Apply Wiener filter
        cleaned_magnitude = magnitude * wiener_gain

        # Reconstruct signal
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = self._inverse_stft(cleaned_stft, len(audio))

        return cleaned_audio

    def noisereduce_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noisereduce library for noise reduction

        Args:
            audio: Input audio chunk

        Returns:
            Cleaned audio
        """
        try:
            # Use noisereduce library
            if self.noise_profile is not None:
                # Stationary noise reduction with profile
                cleaned = nr.reduce_noise(
                    y=audio,
                    sr=self.config.sample_rate,
                    y_noise=self.noise_profile,
                    stationary=self.config.stationary_noise,
                    prop_decrease=self.config.noise_reduction_strength
                )
            else:
                # Non-stationary noise reduction
                cleaned = nr.reduce_noise(
                    y=audio,
                    sr=self.config.sample_rate,
                    stationary=False,
                    prop_decrease=self.config.noise_reduction_strength
                )

            return cleaned

        except Exception as e:
            print(f"Noisereduce error: {e}, returning original audio")
            return audio

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process audio chunk with configured noise cancellation algorithms

        Args:
            audio_chunk: Input audio chunk

        Returns:
            Cleaned audio chunk
        """
        # Flatten if needed
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()

        cleaned = audio_chunk.copy()

        # Apply selected algorithms in sequence
        if self.config.use_spectral_subtraction:
            cleaned = self.spectral_subtraction(cleaned)

        if self.config.use_wiener_filter:
            cleaned = self.wiener_filter(cleaned)

        if self.config.use_rnnoise:
            # RNNoise would require additional setup
            # Placeholder for future implementation
            pass

        # Apply final gain normalization
        cleaned = self._normalize_audio(cleaned, audio_chunk)

        return cleaned

    def _compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform"""
        f, t, stft = signal.stft(
            audio,
            fs=self.config.sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length
        )
        return stft

    def _inverse_stft(self, stft: np.ndarray, length: int) -> np.ndarray:
        """Compute inverse STFT"""
        t, audio = signal.istft(
            stft,
            fs=self.config.sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length
        )

        # Ensure output length matches input
        if len(audio) > length:
            audio = audio[:length]
        elif len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))

        return audio

    def _normalize_audio(self, processed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Normalize processed audio to match original level

        Args:
            processed: Processed audio
            original: Original audio

        Returns:
            Normalized audio
        """
        # Calculate RMS of original and processed
        original_rms = np.sqrt(np.mean(original**2))
        processed_rms = np.sqrt(np.mean(processed**2))

        if processed_rms > 1e-6:
            # Match RMS levels
            gain = original_rms / processed_rms
            # Limit gain to prevent over-amplification
            gain = min(gain, 2.0)
            processed = processed * gain

        # Soft clipping to prevent distortion
        processed = np.tanh(processed)

        return processed

    def reset(self) -> None:
        """Reset noise canceller state"""
        self.noise_profile = None
        self.prev_noise_spectrum = None
        self.smoothed_spectrum = None
        self.noise_psd = None
        self.signal_psd = None
        print("Noise canceller reset")
