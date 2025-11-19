"""
Voice Assistant Module

A comprehensive voice assistant system with:
- Wake word detection
- Speech recognition
- Natural Language Understanding (NLU)
- Text-to-Speech (TTS)
- Plugin system
"""

from .core.assistant import VoiceAssistant

__version__ = "1.0.0"
__all__ = ["VoiceAssistant"]
