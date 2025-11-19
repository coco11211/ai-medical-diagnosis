#!/usr/bin/env python3
"""
Custom Plugin Example

Demonstrates how to create custom plugins for the voice assistant.
"""

import os
import sys
import random
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voice_assistant import VoiceAssistant
from voice_assistant.plugins.base_plugin import BasePlugin


class JokePlugin(BasePlugin):
    """Plugin that tells jokes"""

    def __init__(self, config=None):
        super().__init__(config)
        self.jokes = [
            "Why did the programmer quit his job? Because he didn't get arrays!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "Why do Java developers wear glasses? Because they can't C#!",
            "What's a programmer's favorite place to hang out? The Foo Bar!"
        ]

    def get_intents(self) -> List[str]:
        return ["joke"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        if intent_name == "joke":
            return random.choice(self.jokes)
        return "I don't have a joke for that."

    def initialize(self):
        super().initialize()
        # Add custom intent to NLU if needed
        print(f"{self.name} initialized with {len(self.jokes)} jokes")

    def get_help(self) -> str:
        return "Joke Plugin - Ask me to tell you a joke!"


class CalculatorPlugin(BasePlugin):
    """Plugin that performs calculations"""

    def get_intents(self) -> List[str]:
        return ["calculate"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        # Extract expression from entities or raw text
        expression = entities.get("expression", "")

        if not expression:
            # Try to extract from raw text
            import re
            raw_text = entities.get("raw_text", "")
            match = re.search(r"calculate\s+(.+)", raw_text, re.IGNORECASE)
            if match:
                expression = match.group(1)

        if not expression:
            return "What would you like me to calculate?"

        try:
            # IMPORTANT: In production, use a safe math parser
            # Never use eval() with untrusted input
            # This is for demonstration only
            result = self._safe_calculate(expression)
            return f"The answer is {result}"
        except Exception as e:
            return f"I couldn't calculate that: {str(e)}"

    def _safe_calculate(self, expression: str) -> float:
        """Safely evaluate mathematical expression"""
        import re

        # Only allow numbers and basic operators
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
            raise ValueError("Invalid expression")

        # Use eval (ONLY because we validated the input)
        # In production, use a proper math parser library
        result = eval(expression)
        return result

    def get_help(self) -> str:
        return "Calculator Plugin - Ask me to calculate mathematical expressions"


class QuotePlugin(BasePlugin):
    """Plugin that provides inspirational quotes"""

    def __init__(self, config=None):
        super().__init__(config)
        self.quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Innovation distinguishes between a leader and a follower. - Steve Jobs",
            "Code is like humor. When you have to explain it, it's bad. - Cory House",
            "First, solve the problem. Then, write the code. - John Johnson",
            "Any fool can write code that a computer can understand. Good programmers write code that humans can understand. - Martin Fowler"
        ]

    def get_intents(self) -> List[str]:
        return ["quote", "inspire"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        return random.choice(self.quotes)

    def get_help(self) -> str:
        return "Quote Plugin - Get inspirational quotes"


class WeatherAPIPlugin(BasePlugin):
    """Example plugin that integrates with a weather API"""

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None

    def get_intents(self) -> List[str]:
        return ["weather"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        location = entities.get("location", "your location")

        if not self.api_key:
            return "Weather API key not configured. Please add it to the plugin config."

        # In a real implementation:
        # 1. Call weather API with location and API key
        # 2. Parse response
        # 3. Return formatted weather information

        # For demo purposes:
        return f"Weather in {location}: Sunny, 72°F (This is a demo response. Configure API key for real data.)"

    def get_help(self) -> str:
        return "Weather API Plugin - Get real weather data (requires API key)"


def demo_custom_plugins():
    """Demo custom plugins"""
    print("\n" + "="*60)
    print("Voice Assistant - Custom Plugin Demo")
    print("="*60 + "\n")

    # Create assistant with minimal config for CLI mode
    config = {
        "tts": {
            "engine": "pyttsx3"  # Use pyttsx3 for cross-platform compatibility
        },
        "speak_in_cli": False  # Don't speak in CLI for demo
    }

    assistant = VoiceAssistant(config)

    # Register custom plugins
    print("Registering custom plugins...")
    assistant.add_plugin(JokePlugin())
    assistant.add_plugin(CalculatorPlugin())
    assistant.add_plugin(QuotePlugin())
    assistant.add_plugin(WeatherAPIPlugin({"api_key": None}))

    # First, need to add custom intents to NLU
    print("Adding custom intents to NLU...\n")

    assistant.intent_classifier.add_intent(
        name="joke",
        patterns=[r"\btell.*joke\b", r"\bjoke\b"],
        examples=["tell me a joke", "I want to hear a joke", "joke"]
    )

    assistant.intent_classifier.add_intent(
        name="calculate",
        patterns=[r"\bcalculate\b", r"\bwhat is\s+[\d\+\-\*/]+"],
        examples=["calculate 5 + 3", "what is 10 * 20"],
        entities={"expression": r"calculate\s+(.+)"}
    )

    assistant.intent_classifier.add_intent(
        name="quote",
        patterns=[r"\bquote\b", r"\binspire\b"],
        examples=["give me a quote", "inspire me", "quote"]
    )

    # Test custom plugins
    print("Testing custom plugins:\n")

    test_commands = [
        "tell me a joke",
        "calculate 15 + 27",
        "inspire me",
        "what's the weather in Seattle?",
        "give me another joke"
    ]

    for command in test_commands:
        print(f"User: {command}")
        response = assistant.process_text_command(command)
        print(f"Assistant: {response}")
        print("-" * 60)

    print("\nCustom plugin demo complete!")

    # Show plugin status
    status = assistant.get_status()
    print(f"\nLoaded plugins: {', '.join(status['plugins'])}")
    print(f"Available intents: {', '.join(sorted(status['intents']))}")


def interactive_mode():
    """Run interactive mode with custom plugins"""
    print("\n" + "="*60)
    print("Voice Assistant - Interactive Mode with Custom Plugins")
    print("="*60)
    print("Type commands to interact with custom plugins")
    print("Type 'help' to see available commands")
    print("Type 'quit' to exit")
    print("="*60 + "\n")

    config = {
        "tts": {
            "engine": "pyttsx3"
        },
        "speak_in_cli": False
    }

    assistant = VoiceAssistant(config)

    # Add custom plugins
    assistant.add_plugin(JokePlugin())
    assistant.add_plugin(CalculatorPlugin())
    assistant.add_plugin(QuotePlugin())

    # Add custom intents
    assistant.intent_classifier.add_intent(
        name="joke",
        patterns=[r"\bjoke\b"],
        examples=["tell me a joke"]
    )

    assistant.intent_classifier.add_intent(
        name="calculate",
        patterns=[r"\bcalculate\b"],
        examples=["calculate 5 + 3"],
        entities={"expression": r"calculate\s+(.+)"}
    )

    assistant.intent_classifier.add_intent(
        name="quote",
        patterns=[r"\bquote\b", r"\binspire\b"],
        examples=["give me a quote"]
    )

    print("Custom plugins loaded: JokePlugin, CalculatorPlugin, QuotePlugin")
    print("\nTry commands like:")
    print("  - tell me a joke")
    print("  - calculate 42 * 3")
    print("  - inspire me")
    print("  - what time is it?")
    print()

    assistant.run_cli()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Custom Plugin Example")
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="Run mode (default: demo)"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_custom_plugins()
    else:
        interactive_mode()
