"""
Voice Interface Module
Handles speech recognition and text-to-speech
"""

import speech_recognition as sr
import pyttsx3
from typing import Optional, Dict, Callable
import logging
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceInterface:
    """
    Voice interface for customer service bot
    Supports speech recognition and text-to-speech
    """

    def __init__(
        self,
        language: str = 'en-US',
        voice_rate: int = 150,
        voice_volume: float = 0.9,
        use_background_listening: bool = False
    ):
        """
        Initialize voice interface

        Args:
            language: Language for speech recognition
            voice_rate: Speech rate for TTS (words per minute)
            voice_volume: Volume level (0.0 to 1.0)
            use_background_listening: Enable background listening
        """
        self.language = language
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.use_background_listening = use_background_listening

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = None

        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            self.tts_engine = None

        # Background listening
        self.background_listener = None
        self.listening_active = False
        self.audio_queue = queue.Queue()

        logger.info(f"Voice interface initialized for language: {language}")

    def _configure_tts(self):
        """Configure TTS engine settings"""
        if self.tts_engine:
            # Set rate
            self.tts_engine.setProperty('rate', self.voice_rate)

            # Set volume
            self.tts_engine.setProperty('volume', self.voice_volume)

            # Set voice (prefer female voice on Windows)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                female_voice = next((v for v in voices if 'female' in v.name.lower()), None)
                if female_voice:
                    self.tts_engine.setProperty('voice', female_voice.id)
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)

    def speak(self, text: str, async_mode: bool = False) -> bool:
        """
        Convert text to speech

        Args:
            text: Text to speak
            async_mode: If True, run in background thread

        Returns:
            Success status
        """
        if not self.tts_engine:
            logger.error("TTS engine not initialized")
            return False

        if not text or not text.strip():
            return False

        try:
            if async_mode:
                # Run in background thread
                thread = threading.Thread(target=self._speak_sync, args=(text,))
                thread.daemon = True
                thread.start()
            else:
                # Run synchronously
                self._speak_sync(text)

            return True

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False

    def _speak_sync(self, text: str):
        """Synchronous speech synthesis"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")

    def listen(
        self,
        timeout: Optional[int] = None,
        phrase_time_limit: Optional[int] = None,
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Listen for voice input

        Args:
            timeout: Maximum time to wait for speech (seconds)
            phrase_time_limit: Maximum duration of phrase (seconds)
            prompt: Optional text prompt to speak before listening

        Returns:
            Dictionary with recognition results
        """
        # Speak prompt if provided
        if prompt:
            self.speak(prompt)

        # Initialize microphone if needed
        if self.microphone is None:
            try:
                self.microphone = sr.Microphone()
            except Exception as e:
                logger.error(f"Error initializing microphone: {e}")
                return self._empty_result(error="Microphone not available")

        try:
            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                logger.info("Listening...")

                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            # Recognize speech
            result = self._recognize_speech(audio)
            return result

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return self._empty_result(error="Timeout - no speech detected")

        except Exception as e:
            logger.error(f"Error during listening: {e}")
            return self._empty_result(error=str(e))

    def _recognize_speech(self, audio) -> Dict:
        """
        Recognize speech from audio

        Args:
            audio: Audio data

        Returns:
            Recognition results
        """
        results = {}

        # Try Google Speech Recognition
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            results['google'] = {
                'text': text,
                'success': True,
                'confidence': 0.9  # Google API doesn't provide confidence
            }
            logger.info(f"Recognized (Google): {text}")

        except sr.UnknownValueError:
            results['google'] = {
                'text': '',
                'success': False,
                'error': 'Could not understand audio'
            }
        except sr.RequestError as e:
            results['google'] = {
                'text': '',
                'success': False,
                'error': f'API error: {e}'
            }

        # Try Sphinx (offline) as fallback
        try:
            text = self.recognizer.recognize_sphinx(audio)
            results['sphinx'] = {
                'text': text,
                'success': True,
                'confidence': 0.7  # Lower confidence for offline
            }
            logger.info(f"Recognized (Sphinx): {text}")

        except sr.UnknownValueError:
            results['sphinx'] = {
                'text': '',
                'success': False,
                'error': 'Could not understand audio'
            }
        except sr.RequestError as e:
            results['sphinx'] = {
                'text': '',
                'success': False,
                'error': f'Sphinx error: {e}'
            }
        except Exception:
            # Sphinx might not be installed
            pass

        # Determine best result
        if results.get('google', {}).get('success'):
            best_result = results['google']
            source = 'google'
        elif results.get('sphinx', {}).get('success'):
            best_result = results['sphinx']
            source = 'sphinx'
        else:
            return self._empty_result(error="No recognition successful")

        return {
            'text': best_result['text'],
            'success': True,
            'confidence': best_result.get('confidence', 0.0),
            'source': source,
            'all_results': results
        }

    def _empty_result(self, error: str = "") -> Dict:
        """Return empty recognition result"""
        return {
            'text': '',
            'success': False,
            'confidence': 0.0,
            'error': error,
            'all_results': {}
        }

    def start_background_listening(self, callback: Callable[[Dict], None]):
        """
        Start background listening mode

        Args:
            callback: Function to call with recognition results
        """
        if self.listening_active:
            logger.warning("Background listening already active")
            return

        if self.microphone is None:
            try:
                self.microphone = sr.Microphone()
            except Exception as e:
                logger.error(f"Error initializing microphone: {e}")
                return

        def audio_callback(recognizer, audio):
            """Callback for background audio"""
            try:
                result = self._recognize_speech(audio)
                if result['success']:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in background recognition: {e}")

        try:
            # Start background listening
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            self.background_listener = self.recognizer.listen_in_background(
                self.microphone,
                audio_callback,
                phrase_time_limit=10
            )

            self.listening_active = True
            logger.info("Background listening started")

        except Exception as e:
            logger.error(f"Error starting background listening: {e}")

    def stop_background_listening(self):
        """Stop background listening"""
        if self.background_listener:
            self.background_listener(wait_for_stop=False)
            self.background_listener = None

        self.listening_active = False
        logger.info("Background listening stopped")

    def set_voice_rate(self, rate: int):
        """Set speech rate"""
        if self.tts_engine:
            self.voice_rate = rate
            self.tts_engine.setProperty('rate', rate)
            logger.info(f"Voice rate set to {rate}")

    def set_voice_volume(self, volume: float):
        """Set speech volume"""
        if self.tts_engine:
            self.voice_volume = max(0.0, min(1.0, volume))
            self.tts_engine.setProperty('volume', self.voice_volume)
            logger.info(f"Voice volume set to {self.voice_volume}")

    def set_voice(self, voice_id: Optional[int] = None, gender: Optional[str] = None):
        """
        Set TTS voice

        Args:
            voice_id: Specific voice ID
            gender: 'male' or 'female' to select voice by gender
        """
        if not self.tts_engine:
            return

        voices = self.tts_engine.getProperty('voices')

        if voice_id is not None and 0 <= voice_id < len(voices):
            self.tts_engine.setProperty('voice', voices[voice_id].id)
            logger.info(f"Voice set to: {voices[voice_id].name}")

        elif gender:
            matching_voice = next(
                (v for v in voices if gender.lower() in v.name.lower()),
                None
            )
            if matching_voice:
                self.tts_engine.setProperty('voice', matching_voice.id)
                logger.info(f"Voice set to: {matching_voice.name}")

    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if self.tts_engine:
            voices = self.tts_engine.getProperty('voices')
            return [
                {
                    'id': i,
                    'name': v.name,
                    'languages': v.languages,
                    'gender': v.gender
                }
                for i, v in enumerate(voices)
            ]
        return []

    def get_microphone_info(self) -> Dict:
        """Get information about available microphones"""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            return {
                'count': len(mic_list),
                'devices': [
                    {'id': i, 'name': name}
                    for i, name in enumerate(mic_list)
                ]
            }
        except Exception as e:
            logger.error(f"Error getting microphone info: {e}")
            return {'count': 0, 'devices': []}

    def set_microphone(self, device_index: Optional[int] = None):
        """Set microphone device"""
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            logger.info(f"Microphone set to device index: {device_index}")
        except Exception as e:
            logger.error(f"Error setting microphone: {e}")

    def test_voice(self):
        """Test voice output with sample text"""
        test_text = "Hello! This is a test of the customer service voice interface."
        self.speak(test_text)

    def test_recognition(self):
        """Test speech recognition"""
        print("Testing speech recognition...")
        print("Please speak into the microphone...")

        result = self.listen(timeout=5, phrase_time_limit=10)

        if result['success']:
            print(f"Recognized: {result['text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Source: {result['source']}")

            # Echo back
            self.speak(f"You said: {result['text']}")
        else:
            print(f"Recognition failed: {result.get('error', 'Unknown error')}")


# Testing function
def test_voice_interface():
    """Test voice interface"""
    print("Voice Interface Test")
    print("=" * 70)

    vi = VoiceInterface()

    # Test TTS
    print("\n1. Testing Text-to-Speech")
    print("-" * 70)
    test_phrases = [
        "Hello! Welcome to our customer service.",
        "How can I help you today?",
        "Your request is being processed."
    ]

    for phrase in test_phrases:
        print(f"Speaking: {phrase}")
        vi.speak(phrase)

    # Show available voices
    print("\n2. Available Voices")
    print("-" * 70)
    voices = vi.get_available_voices()
    for voice in voices:
        print(f"  {voice['id']}: {voice['name']}")

    # Show microphones
    print("\n3. Available Microphones")
    print("-" * 70)
    mics = vi.get_microphone_info()
    for mic in mics['devices']:
        print(f"  {mic['id']}: {mic['name']}")

    # Test recognition (commented out for non-interactive testing)
    # print("\n4. Testing Speech Recognition")
    # print("-" * 70)
    # vi.test_recognition()


if __name__ == "__main__":
    test_voice_interface()
