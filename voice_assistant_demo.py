#!/usr/bin/env python3
"""
Voice Assistant Demo

Demonstrates the voice assistant functionality with various modes.
"""

import os
import sys
import argparse
import logging
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voice_assistant import VoiceAssistant
from voice_assistant.plugins.base_plugin import BasePlugin


class CustomPlugin(BasePlugin):
    """Example custom plugin"""

    def get_intents(self):
        return ["custom_intent"]

    def handle(self, intent_name, entities, context):
        return "This is a custom plugin response!"


def load_config(config_file: str = "voice_assistant_config.yaml"):
    """Load configuration from YAML file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_voice_mode(config):
    """Run in voice-activated mode"""
    print("\n" + "="*60)
    print("Voice Assistant - Voice Mode")
    print("="*60)
    print("Say the wake word to activate the assistant")
    print("Wake words:", config.get("wake_word", {}).get("words", ["assistant"]))
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    assistant = VoiceAssistant(config)

    # Optionally add custom plugin
    # assistant.add_plugin(CustomPlugin())

    assistant.run()


def run_cli_mode(config):
    """Run in CLI (text-based) mode"""
    assistant = VoiceAssistant(config)

    # Optionally add custom plugin
    # assistant.add_plugin(CustomPlugin())

    assistant.run_cli()


def run_demo_mode(config):
    """Run demo mode with sample commands"""
    print("\n" + "="*60)
    print("Voice Assistant - Demo Mode")
    print("="*60)
    print("Processing sample commands...")
    print("="*60 + "\n")

    assistant = VoiceAssistant(config)

    # Sample commands
    commands = [
        "Hello",
        "What time is it?",
        "Set a timer for 5 minutes",
        "Remind me to call John tomorrow",
        "What's the weather?",
        "Show me the stock price of AAPL",
        "Search for Python tutorials",
        "Help",
        "Goodbye"
    ]

    for command in commands:
        print(f"\nUser: {command}")
        response = assistant.process_text_command(command)
        print(f"Assistant: {response}")
        print("-" * 60)

        # Optionally speak the response
        if config.get("speak_in_cli", False):
            assistant.tts.speak(response, blocking=True)

    print("\nDemo complete!")


def test_components(config):
    """Test individual components"""
    print("\n" + "="*60)
    print("Voice Assistant - Component Test")
    print("="*60 + "\n")

    # Test TTS
    print("Testing Text-to-Speech...")
    from voice_assistant.tts.speech_synthesizer import SpeechSynthesizer

    tts_config = config.get("tts", {})
    tts = SpeechSynthesizer(
        engine=tts_config.get("engine", "pyttsx3"),
        rate=tts_config.get("rate", 150),
        volume=tts_config.get("volume", 1.0)
    )

    print("Available voices:")
    voices = tts.get_voices()
    for i, voice in enumerate(voices[:5]):  # Show first 5 voices
        print(f"  {i+1}. {voice}")

    tts.speak("Hello! Voice assistant component test.", blocking=True)
    print("TTS test complete.\n")

    # Test Speech Recognition
    print("Testing Speech Recognition...")
    print("Say something (you have 5 seconds)...")
    from voice_assistant.speech.recognizer import SpeechRecognizer

    speech_config = config.get("speech", {})
    recognizer = SpeechRecognizer(
        engine=speech_config.get("engine", "google"),
        language=speech_config.get("language", "en")
    )

    text = recognizer.listen_and_recognize()
    if text:
        print(f"Recognized: {text}")
    else:
        print("No speech recognized")
    print("Speech recognition test complete.\n")

    # Test NLU
    print("Testing Natural Language Understanding...")
    from voice_assistant.nlu.intent_classifier import IntentClassifier

    nlu = IntentClassifier(backend="pattern")

    test_phrases = [
        "What time is it?",
        "Set a timer for 10 minutes",
        "What's the weather in New York?",
        "Goodbye"
    ]

    for phrase in test_phrases:
        intent = nlu.classify(phrase)
        print(f"  '{phrase}'")
        print(f"    -> Intent: {intent.name} (confidence: {intent.confidence:.2f})")
        print(f"    -> Entities: {intent.entities}")

    print("\nNLU test complete.\n")

    print("Component testing complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Voice Assistant Demo")
    parser.add_argument(
        "--mode",
        choices=["voice", "cli", "demo", "test"],
        default="cli",
        help="Operation mode (default: cli)"
    )
    parser.add_argument(
        "--config",
        default="voice_assistant_config.yaml",
        help="Configuration file (default: voice_assistant_config.yaml)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = load_config(args.config)

    # Override logging level from config
    if not args.debug and "logging_level" in config:
        logging.getLogger().setLevel(config["logging_level"])

    # Run in selected mode
    try:
        if args.mode == "voice":
            run_voice_mode(config)
        elif args.mode == "cli":
            run_cli_mode(config)
        elif args.mode == "demo":
            run_demo_mode(config)
        elif args.mode == "test":
            test_components(config)

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
