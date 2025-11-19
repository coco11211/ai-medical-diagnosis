"""
Voice Assistant Core

Main orchestrator that coordinates all components:
- Wake word detection
- Speech recognition
- Natural Language Understanding
- Text-to-Speech
- Plugin management
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, Callable
import signal
import sys

from ..wake_word.detector import WakeWordDetector
from ..speech.recognizer import SpeechRecognizer
from ..nlu.intent_classifier import IntentClassifier, Intent
from ..tts.speech_synthesizer import SpeechSynthesizer
from ..plugins.base_plugin import PluginManager
from ..plugins.builtin_plugins import (
    TimePlugin,
    WeatherPlugin,
    TimerPlugin,
    ReminderPlugin,
    NewsPlugin,
    TradingPlugin,
    GreetingPlugin,
    HelpPlugin,
    SearchPlugin
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Main voice assistant orchestrator"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize voice assistant

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_running = False
        self.is_listening_for_command = False

        # Initialize components
        self._initialize_components()

        # Context for plugins
        self.context = {
            "tts": self.tts,
            "plugin_manager": self.plugin_manager,
            "assistant": self
        }

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_components(self):
        """Initialize all voice assistant components"""
        logger.info("Initializing voice assistant components...")

        # Wake word detector
        wake_word_config = self.config.get("wake_word", {})
        self.wake_word_detector = WakeWordDetector(
            wake_words=wake_word_config.get("words", ["assistant"]),
            sensitivity=wake_word_config.get("sensitivity", 0.5),
            engine=wake_word_config.get("engine", "simple"),
            access_key=wake_word_config.get("access_key"),
            callback=self._on_wake_word_detected
        )

        # Speech recognizer
        speech_config = self.config.get("speech", {})
        self.speech_recognizer = SpeechRecognizer(
            engine=speech_config.get("engine", "google"),
            language=speech_config.get("language", "en"),
            model_size=speech_config.get("model_size", "base"),
            device=speech_config.get("device", "cpu"),
            timeout=speech_config.get("timeout", 5.0),
            phrase_timeout=speech_config.get("phrase_timeout", 3.0)
        )

        # NLU intent classifier
        nlu_config = self.config.get("nlu", {})
        self.intent_classifier = IntentClassifier(
            backend=nlu_config.get("backend", "pattern"),
            model_path=nlu_config.get("model_path"),
            intents_config=nlu_config.get("intents")
        )

        # Text-to-Speech
        tts_config = self.config.get("tts", {})
        self.tts = SpeechSynthesizer(
            engine=tts_config.get("engine", "sapi"),
            voice=tts_config.get("voice"),
            rate=tts_config.get("rate", 150),
            volume=tts_config.get("volume", 1.0),
            language=tts_config.get("language", "en-US")
        )

        # Plugin manager
        self.plugin_manager = PluginManager()
        self._register_builtin_plugins()

        # Load custom plugins if directory specified
        plugin_dir = self.config.get("plugin_directory")
        if plugin_dir:
            self.plugin_manager.load_from_directory(plugin_dir)

        logger.info("All components initialized successfully")

    def _register_builtin_plugins(self):
        """Register built-in plugins"""
        builtin_plugins = [
            GreetingPlugin(),
            TimePlugin(),
            WeatherPlugin(),
            TimerPlugin(),
            ReminderPlugin(),
            NewsPlugin(),
            TradingPlugin(),
            HelpPlugin(),
            SearchPlugin()
        ]

        for plugin in builtin_plugins:
            self.plugin_manager.register(plugin)

    def _on_wake_word_detected(self, wake_word: str):
        """Callback when wake word is detected"""
        if self.is_listening_for_command:
            logger.info("Already listening for command, ignoring wake word")
            return

        logger.info(f"Wake word detected: {wake_word}")
        self.is_listening_for_command = True

        # Acknowledge wake word
        self.tts.speak("Yes?", blocking=False)

        # Listen for command in separate thread
        threading.Thread(target=self._listen_for_command, daemon=True).start()

    def _listen_for_command(self):
        """Listen for and process voice command"""
        try:
            # Recognize speech
            logger.info("Listening for command...")
            text = self.speech_recognizer.listen_and_recognize()

            if text:
                logger.info(f"Recognized: {text}")
                self._process_command(text)
            else:
                logger.warning("No speech recognized")
                self.tts.speak("I didn't catch that. Please try again.", blocking=False)

        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            self.tts.speak("Sorry, I encountered an error.", blocking=False)

        finally:
            self.is_listening_for_command = False

    def _process_command(self, text: str):
        """Process voice command"""
        try:
            # Classify intent
            intent = self.intent_classifier.classify(text)
            logger.info(f"Intent: {intent.name} (confidence: {intent.confidence:.2f})")
            logger.info(f"Entities: {intent.entities}")

            # Handle with plugin
            response = self.plugin_manager.handle_intent(
                intent.name,
                intent.entities,
                self.context
            )

            if response:
                logger.info(f"Response: {response}")
                self.tts.speak(response, blocking=False)
            else:
                # No plugin could handle it
                self.tts.speak(
                    "I'm not sure how to help with that. Try asking for help to see what I can do.",
                    blocking=False
                )

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.tts.speak("Sorry, I had trouble processing that request.", blocking=False)

    def process_text_command(self, text: str) -> str:
        """
        Process text command (without voice input)

        Args:
            text: Command text

        Returns:
            Response text
        """
        intent = self.intent_classifier.classify(text)
        logger.info(f"Intent: {intent.name} (confidence: {intent.confidence:.2f})")

        response = self.plugin_manager.handle_intent(
            intent.name,
            intent.entities,
            self.context
        )

        return response or "I'm not sure how to help with that."

    def start(self):
        """Start the voice assistant"""
        if self.is_running:
            logger.warning("Voice assistant already running")
            return

        logger.info("Starting voice assistant...")
        self.is_running = True

        # Start wake word detection
        self.wake_word_detector.start()

        logger.info("Voice assistant started. Listening for wake word...")
        self.tts.speak("Voice assistant ready", blocking=False)

    def stop(self):
        """Stop the voice assistant"""
        if not self.is_running:
            return

        logger.info("Stopping voice assistant...")
        self.is_running = False

        # Stop wake word detection
        self.wake_word_detector.stop()

        # Shutdown plugins
        self.plugin_manager.shutdown_all()

        logger.info("Voice assistant stopped")

    def run(self):
        """Run the voice assistant (blocking)"""
        self.start()

        try:
            # Keep running until interrupted
            while self.is_running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.stop()

    def run_cli(self):
        """Run in CLI mode (text-based interaction)"""
        logger.info("Starting voice assistant in CLI mode...")
        self.tts.speak("Voice assistant started in CLI mode", blocking=False)

        print("\n" + "="*60)
        print("Voice Assistant - CLI Mode")
        print("="*60)
        print("Type 'help' for available commands")
        print("Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")

        try:
            while True:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    response = "Goodbye!"
                    print(f"Assistant: {response}")
                    self.tts.speak(response, blocking=True)
                    break

                # Process command
                response = self.process_text_command(user_input)
                print(f"Assistant: {response}")

                # Optionally speak response
                if self.config.get("speak_in_cli", False):
                    self.tts.speak(response, blocking=False)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("\nGoodbye!")

        except Exception as e:
            logger.error(f"Error in CLI mode: {e}")

        finally:
            self.plugin_manager.shutdown_all()

    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def add_plugin(self, plugin):
        """
        Add a custom plugin

        Args:
            plugin: Plugin instance
        """
        self.plugin_manager.register(plugin)

    def get_status(self) -> Dict[str, Any]:
        """Get assistant status"""
        return {
            "is_running": self.is_running,
            "is_listening": self.is_listening_for_command,
            "plugins": self.plugin_manager.list_plugins(),
            "intents": self.plugin_manager.get_all_intents()
        }
