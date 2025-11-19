"""
Text-to-Speech (TTS) Module

Supports multiple TTS engines:
1. Windows SAPI (Native Windows 11 support)
2. pyttsx3 (Cross-platform, offline)
3. Google TTS (Cloud-based, high quality)
4. Edge TTS (Microsoft Edge TTS, free)
"""

import os
import logging
import platform
from typing import Optional, List
import tempfile
import threading

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Text-to-Speech with multiple engine support"""

    def __init__(
        self,
        engine: str = "sapi",
        voice: Optional[str] = None,
        rate: int = 150,
        volume: float = 1.0,
        language: str = "en-US"
    ):
        """
        Initialize speech synthesizer

        Args:
            engine: TTS engine ("sapi", "pyttsx3", "gtts", "edge")
            voice: Voice name/ID (engine-specific)
            rate: Speech rate (words per minute, typically 100-200)
            volume: Volume level (0.0 to 1.0)
            language: Language code (e.g., "en-US", "es-ES")
        """
        self.engine_name = engine
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.language = language

        self.engine = None
        self.is_speaking = False

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the selected TTS engine"""
        try:
            if self.engine_name == "sapi":
                self._initialize_sapi()
            elif self.engine_name == "pyttsx3":
                self._initialize_pyttsx3()
            elif self.engine_name == "gtts":
                logger.info("Google TTS initialized (requires internet)")
            elif self.engine_name == "edge":
                logger.info("Edge TTS initialized (requires internet)")
            else:
                logger.warning(f"Unknown engine '{self.engine_name}', using pyttsx3")
                self.engine_name = "pyttsx3"
                self._initialize_pyttsx3()
        except Exception as e:
            logger.error(f"Failed to initialize {self.engine_name}: {e}")
            logger.info("Falling back to pyttsx3")
            self.engine_name = "pyttsx3"
            self._initialize_pyttsx3()

    def _initialize_sapi(self):
        """Initialize Windows SAPI TTS"""
        if platform.system() != "Windows":
            raise OSError("SAPI is only available on Windows")

        try:
            import win32com.client

            self.engine = win32com.client.Dispatch("SAPI.SpVoice")

            # Set voice if specified
            if self.voice:
                voices = self.engine.GetVoices()
                for i in range(voices.Count):
                    if self.voice.lower() in voices.Item(i).GetDescription().lower():
                        self.engine.Voice = voices.Item(i)
                        break

            # Set rate (-10 to 10, default 0)
            # Convert from WPM to SAPI rate
            sapi_rate = max(-10, min(10, (self.rate - 150) // 10))
            self.engine.Rate = sapi_rate

            # Set volume (0 to 100)
            self.engine.Volume = int(self.volume * 100)

            logger.info(f"Windows SAPI initialized with voice: {self.engine.Voice.GetDescription()}")

        except ImportError:
            raise ImportError("pywin32 not installed. Install with: pip install pywin32")

    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 TTS"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Set voice if specified
            if self.voice:
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if self.voice.lower() in v.name.lower() or self.voice == v.id:
                        self.engine.setProperty('voice', v.id)
                        break

            # Set rate
            self.engine.setProperty('rate', self.rate)

            # Set volume
            self.engine.setProperty('volume', self.volume)

            logger.info("pyttsx3 initialized")

        except ImportError:
            raise ImportError("pyttsx3 not installed. Install with: pip install pyttsx3")

    def speak(self, text: str, blocking: bool = True):
        """
        Speak the given text

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if not text:
            logger.warning("Empty text provided to speak()")
            return

        try:
            if self.engine_name == "sapi":
                self._speak_sapi(text, blocking)
            elif self.engine_name == "pyttsx3":
                self._speak_pyttsx3(text, blocking)
            elif self.engine_name == "gtts":
                self._speak_gtts(text, blocking)
            elif self.engine_name == "edge":
                self._speak_edge(text, blocking)

        except Exception as e:
            logger.error(f"Error speaking text: {e}")

    def _speak_sapi(self, text: str, blocking: bool):
        """Speak using Windows SAPI"""
        import win32com.client

        flags = 0 if blocking else 1  # 0 = synchronous, 1 = asynchronous
        self.is_speaking = True
        self.engine.Speak(text, flags)
        self.is_speaking = False

    def _speak_pyttsx3(self, text: str, blocking: bool):
        """Speak using pyttsx3"""
        self.is_speaking = True

        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
        else:
            def speak_async():
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False

            thread = threading.Thread(target=speak_async)
            thread.daemon = True
            thread.start()

    def _speak_gtts(self, text: str, blocking: bool):
        """Speak using Google TTS"""
        try:
            from gtts import gTTS
            import pygame

            # Generate speech
            tts = gTTS(text=text, lang=self.language[:2], slow=False)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                temp_file = f.name
                tts.save(temp_file)

            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.set_volume(self.volume)

            self.is_speaking = True
            pygame.mixer.music.play()

            if blocking:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                os.unlink(temp_file)
                self.is_speaking = False
            else:
                def cleanup():
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                    os.unlink(temp_file)
                    self.is_speaking = False

                thread = threading.Thread(target=cleanup)
                thread.daemon = True
                thread.start()

        except ImportError:
            logger.error("gtts or pygame not installed. Install with: pip install gtts pygame")
        except Exception as e:
            logger.error(f"Google TTS error: {e}")

    def _speak_edge(self, text: str, blocking: bool):
        """Speak using Edge TTS"""
        try:
            import edge_tts
            import asyncio
            import pygame

            async def generate_speech():
                communicate = edge_tts.Communicate(text, self.language)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    temp_file = f.name
                    await communicate.save(temp_file)
                return temp_file

            # Generate speech
            temp_file = asyncio.run(generate_speech())

            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.set_volume(self.volume)

            self.is_speaking = True
            pygame.mixer.music.play()

            if blocking:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                os.unlink(temp_file)
                self.is_speaking = False
            else:
                def cleanup():
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                    os.unlink(temp_file)
                    self.is_speaking = False

                thread = threading.Thread(target=cleanup)
                thread.daemon = True
                thread.start()

        except ImportError:
            logger.error("edge-tts or pygame not installed. Install with: pip install edge-tts pygame")
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")

    def stop(self):
        """Stop current speech"""
        try:
            if self.engine_name == "sapi":
                import win32com.client
                self.engine.Speak("", 3)  # Flag 3 = purge before speak
            elif self.engine_name == "pyttsx3":
                self.engine.stop()
            elif self.engine_name in ["gtts", "edge"]:
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()

            self.is_speaking = False
            logger.info("Speech stopped")

        except Exception as e:
            logger.error(f"Error stopping speech: {e}")

    def get_voices(self) -> List[str]:
        """Get list of available voices"""
        voices = []

        try:
            if self.engine_name == "sapi":
                import win32com.client
                voice_list = self.engine.GetVoices()
                for i in range(voice_list.Count):
                    voices.append(voice_list.Item(i).GetDescription())

            elif self.engine_name == "pyttsx3":
                voice_list = self.engine.getProperty('voices')
                voices = [v.name for v in voice_list]

            elif self.engine_name == "edge":
                import edge_tts
                import asyncio

                async def get_edge_voices():
                    return await edge_tts.list_voices()

                voice_list = asyncio.run(get_edge_voices())
                voices = [f"{v['Name']} ({v['Locale']})" for v in voice_list]

        except Exception as e:
            logger.error(f"Error getting voices: {e}")

        return voices

    def set_voice(self, voice: str):
        """Change the current voice"""
        self.voice = voice

        try:
            if self.engine_name == "sapi":
                import win32com.client
                voices = self.engine.GetVoices()
                for i in range(voices.Count):
                    if voice.lower() in voices.Item(i).GetDescription().lower():
                        self.engine.Voice = voices.Item(i)
                        logger.info(f"Voice changed to: {voice}")
                        return

            elif self.engine_name == "pyttsx3":
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower() or voice == v.id:
                        self.engine.setProperty('voice', v.id)
                        logger.info(f"Voice changed to: {voice}")
                        return

        except Exception as e:
            logger.error(f"Error setting voice: {e}")

    def set_rate(self, rate: int):
        """Change speech rate"""
        self.rate = rate

        try:
            if self.engine_name == "sapi":
                sapi_rate = max(-10, min(10, (rate - 150) // 10))
                self.engine.Rate = sapi_rate
            elif self.engine_name == "pyttsx3":
                self.engine.setProperty('rate', rate)

        except Exception as e:
            logger.error(f"Error setting rate: {e}")

    def set_volume(self, volume: float):
        """Change volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))

        try:
            if self.engine_name == "sapi":
                self.engine.Volume = int(self.volume * 100)
            elif self.engine_name == "pyttsx3":
                self.engine.setProperty('volume', self.volume)

        except Exception as e:
            logger.error(f"Error setting volume: {e}")
