"""
Speech Recognition Module

Supports multiple speech recognition engines:
1. Faster Whisper (OpenAI Whisper optimized) - High accuracy, runs locally
2. Google Speech Recognition - Cloud-based
3. Windows Speech Recognition (SAPI) - Native Windows support
"""

import os
import logging
import numpy as np
import pyaudio
import wave
import tempfile
from typing import Optional, Dict
import speech_recognition as sr

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Speech recognition with multiple engine support"""

    def __init__(
        self,
        engine: str = "whisper",
        language: str = "en",
        model_size: str = "base",
        device: str = "cpu",
        timeout: float = 5.0,
        phrase_timeout: float = 3.0
    ):
        """
        Initialize speech recognizer

        Args:
            engine: Recognition engine ("whisper", "google", "windows")
            language: Language code (e.g., "en", "es", "fr")
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device for processing ("cpu", "cuda")
            timeout: Maximum time to wait for speech (seconds)
            phrase_timeout: Maximum time of silence before ending (seconds)
        """
        self.engine = engine
        self.language = language
        self.model_size = model_size
        self.device = device
        self.timeout = timeout
        self.phrase_timeout = phrase_timeout

        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Whisper model
        self.whisper_model = None

        # Adjust for ambient noise
        with self.microphone as source:
            logger.info("Calibrating for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the selected speech recognition engine"""
        try:
            if self.engine == "whisper":
                self._initialize_whisper()
            elif self.engine == "google":
                logger.info("Using Google Speech Recognition")
            elif self.engine == "windows":
                logger.info("Using Windows Speech Recognition (SAPI)")
            else:
                logger.warning(f"Unknown engine '{self.engine}', using Google")
                self.engine = "google"
        except Exception as e:
            logger.error(f"Failed to initialize {self.engine} engine: {e}")
            logger.info("Falling back to Google Speech Recognition")
            self.engine = "google"

    def _initialize_whisper(self):
        """Initialize Faster Whisper model"""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16"
            )
            logger.info("Whisper model loaded successfully")

        except ImportError:
            raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")

    def listen(self) -> Optional[sr.AudioData]:
        """
        Listen for speech and return audio data

        Returns:
            AudioData object or None if no speech detected
        """
        try:
            with self.microphone as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_timeout
                )
                return audio

        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout")
            return None
        except Exception as e:
            logger.error(f"Error while listening: {e}")
            return None

    def recognize(self, audio: Optional[sr.AudioData] = None) -> Optional[str]:
        """
        Recognize speech from audio data

        Args:
            audio: Audio data (if None, will listen first)

        Returns:
            Recognized text or None if recognition failed
        """
        if audio is None:
            audio = self.listen()
            if audio is None:
                return None

        try:
            if self.engine == "whisper":
                return self._recognize_whisper(audio)
            elif self.engine == "google":
                return self._recognize_google(audio)
            elif self.engine == "windows":
                return self._recognize_windows(audio)
            else:
                return self._recognize_google(audio)

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

    def _recognize_whisper(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using Faster Whisper"""
        try:
            # Convert audio to wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
                with wave.open(wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(audio.sample_rate)
                    wav_file.writeframes(audio.get_wav_data())

            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                wav_path,
                language=self.language if self.language != "auto" else None,
                beam_size=5,
                vad_filter=True
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()

            # Clean up temp file
            os.unlink(wav_path)

            logger.info(f"Recognized (Whisper): {text}")
            return text if text else None

        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            return None

    def _recognize_google(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(
                audio,
                language=self.language
            )
            logger.info(f"Recognized (Google): {text}")
            return text

        except sr.UnknownValueError:
            logger.warning("Google could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google API error: {e}")
            return None

    def _recognize_windows(self, audio: sr.AudioData) -> Optional[str]:
        """Recognize speech using Windows Speech Recognition"""
        try:
            # Windows Speech Recognition via SAPI
            text = self.recognizer.recognize_sphinx(audio)
            logger.info(f"Recognized (Windows): {text}")
            return text

        except sr.UnknownValueError:
            logger.warning("Windows SAPI could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Windows SAPI error: {e}")
            return None
        except Exception as e:
            # Fallback to Google if Windows SAPI not available
            logger.warning(f"Windows SAPI not available: {e}, falling back to Google")
            return self._recognize_google(audio)

    def listen_and_recognize(self) -> Optional[str]:
        """
        Convenience method to listen and recognize in one call

        Returns:
            Recognized text or None
        """
        audio = self.listen()
        if audio:
            return self.recognize(audio)
        return None

    def set_energy_threshold(self, threshold: int):
        """Set energy threshold for voice detection"""
        self.recognizer.energy_threshold = threshold
        logger.info(f"Energy threshold set to {threshold}")

    def set_dynamic_energy_threshold(self, enabled: bool = True):
        """Enable/disable dynamic energy threshold adjustment"""
        self.recognizer.dynamic_energy_threshold = enabled
        logger.info(f"Dynamic energy threshold: {enabled}")
