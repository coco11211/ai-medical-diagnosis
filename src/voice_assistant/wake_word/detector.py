"""
Wake Word Detection using Porcupine and OpenWakeWord

Supports multiple wake word detection engines:
1. Porcupine (Picovoice) - High accuracy, requires API key
2. OpenWakeWord - Open source, runs locally
3. Simple energy-based detection (fallback)
"""

import os
import struct
import pyaudio
import numpy as np
from typing import Optional, Callable, List
import logging
import threading

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Wake word detection with multiple engine support"""

    def __init__(
        self,
        wake_words: List[str] = None,
        sensitivity: float = 0.5,
        engine: str = "openwakeword",
        access_key: Optional[str] = None,
        callback: Optional[Callable] = None
    ):
        """
        Initialize wake word detector

        Args:
            wake_words: List of wake words (e.g., ["jarvis", "alexa"])
            sensitivity: Detection sensitivity (0.0-1.0)
            engine: Detection engine ("porcupine", "openwakeword", "simple")
            access_key: Porcupine API access key (required for porcupine engine)
            callback: Callback function when wake word detected
        """
        self.wake_words = wake_words or ["assistant"]
        self.sensitivity = sensitivity
        self.engine = engine
        self.access_key = access_key
        self.callback = callback
        self.is_running = False
        self.audio_stream = None
        self.pa = None
        self.detector = None

        # Audio configuration
        self.sample_rate = 16000
        self.frame_length = 512
        self.channels = 1

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the selected wake word detection engine"""
        try:
            if self.engine == "porcupine":
                self._initialize_porcupine()
            elif self.engine == "openwakeword":
                self._initialize_openwakeword()
            elif self.engine == "simple":
                self._initialize_simple()
            else:
                logger.warning(f"Unknown engine '{self.engine}', using simple detector")
                self.engine = "simple"
                self._initialize_simple()
        except Exception as e:
            logger.error(f"Failed to initialize {self.engine} detector: {e}")
            logger.info("Falling back to simple energy-based detector")
            self.engine = "simple"
            self._initialize_simple()

    def _initialize_porcupine(self):
        """Initialize Porcupine wake word detector"""
        try:
            import pvporcupine

            if not self.access_key:
                raise ValueError("Porcupine requires an access_key")

            # Use built-in wake words or custom models
            keywords = []
            keyword_paths = []

            for wake_word in self.wake_words:
                wake_word_lower = wake_word.lower()
                # Check if it's a built-in keyword
                if wake_word_lower in pvporcupine.KEYWORDS:
                    keywords.append(wake_word_lower)
                else:
                    # For custom keywords, you'd need .ppn files
                    logger.warning(f"Wake word '{wake_word}' not in built-in keywords")

            if not keywords and not keyword_paths:
                logger.warning("No valid wake words, using 'porcupine' as default")
                keywords = ["porcupine"]

            self.detector = pvporcupine.create(
                access_key=self.access_key,
                keywords=keywords,
                sensitivities=[self.sensitivity] * len(keywords)
            )
            self.sample_rate = self.detector.sample_rate
            self.frame_length = self.detector.frame_length

            logger.info(f"Porcupine initialized with keywords: {keywords}")

        except ImportError:
            raise ImportError("pvporcupine not installed. Install with: pip install pvporcupine")

    def _initialize_openwakeword(self):
        """Initialize OpenWakeWord detector"""
        try:
            from openwakeword.model import Model

            # OpenWakeWord uses pre-trained models
            self.detector = Model(
                wakeword_models=[],  # Uses default models
                inference_framework='onnx'
            )

            # Get available models
            available_models = self.detector.models.keys()
            logger.info(f"OpenWakeWord initialized with models: {list(available_models)}")

            # Sample rate for OpenWakeWord
            self.sample_rate = 16000
            self.frame_length = 1280  # 80ms at 16kHz

        except ImportError:
            raise ImportError("openwakeword not installed. Install with: pip install openwakeword")

    def _initialize_simple(self):
        """Initialize simple energy-based detector (fallback)"""
        self.detector = SimpleEnergyDetector(
            wake_words=self.wake_words,
            sensitivity=self.sensitivity
        )
        logger.info("Simple energy-based detector initialized")

    def start(self):
        """Start listening for wake words"""
        if self.is_running:
            logger.warning("Wake word detector already running")
            return

        self.is_running = True
        self.pa = pyaudio.PyAudio()

        try:
            self.audio_stream = self.pa.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.frame_length,
                stream_callback=self._audio_callback
            )

            self.audio_stream.start_stream()
            logger.info(f"Wake word detection started (engine: {self.engine})")

        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.is_running = False
            raise

    def stop(self):
        """Stop listening for wake words"""
        if not self.is_running:
            return

        self.is_running = False

        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None

        if self.pa:
            self.pa.terminate()
            self.pa = None

        logger.info("Wake word detection stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if not self.is_running:
            return (None, pyaudio.paComplete)

        try:
            pcm = struct.unpack_from("h" * frame_count, in_data)

            detected = False
            wake_word = None

            if self.engine == "porcupine":
                keyword_index = self.detector.process(pcm)
                if keyword_index >= 0:
                    detected = True
                    wake_word = self.wake_words[keyword_index] if keyword_index < len(self.wake_words) else "unknown"

            elif self.engine == "openwakeword":
                # OpenWakeWord expects numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                prediction = self.detector.predict(audio_data)

                # Check if any model triggered
                for model_name, score in prediction.items():
                    if score > self.sensitivity:
                        detected = True
                        wake_word = model_name
                        break

            elif self.engine == "simple":
                detected, wake_word = self.detector.process(pcm)

            if detected:
                logger.info(f"Wake word detected: {wake_word}")
                if self.callback:
                    # Run callback in separate thread to avoid blocking audio
                    threading.Thread(target=self.callback, args=(wake_word,)).start()

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

        return (in_data, pyaudio.paContinue)

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class SimpleEnergyDetector:
    """Simple energy-based wake word detector (fallback)"""

    def __init__(self, wake_words: List[str], sensitivity: float = 0.5):
        self.wake_words = wake_words
        self.sensitivity = sensitivity
        self.energy_threshold = 1000 * (1 - sensitivity)
        self.consecutive_frames = 0
        self.required_frames = 3

    def process(self, pcm):
        """Process audio frame"""
        # Calculate energy
        energy = np.sqrt(np.mean(np.square(pcm)))

        if energy > self.energy_threshold:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.required_frames:
                self.consecutive_frames = 0
                # Return first wake word as detected
                return True, self.wake_words[0]
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        return False, None
