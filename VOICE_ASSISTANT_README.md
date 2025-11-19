# Voice Assistant for Windows 11

A comprehensive voice assistant system with wake word detection, speech recognition, natural language understanding (NLU), text-to-speech (TTS), and a plugin architecture.

## Features

- **Wake Word Detection**: Multiple engines (Porcupine, OpenWakeWord, Simple energy-based)
- **Speech Recognition**: Supports Faster Whisper (local), Google Cloud, Windows SAPI
- **NLU**: Pattern-based, scikit-learn, or transformer-based intent classification
- **TTS**: Windows SAPI (native Windows 11), pyttsx3, Google TTS, Edge TTS
- **Plugin System**: Extensible architecture for adding custom functionality
- **Built-in Plugins**:
  - Time & Date
  - Weather queries
  - Timers & Reminders
  - News headlines
  - Web search
  - Trading bot integration
  - Help system

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Windows-Specific Setup

For Windows 11 with SAPI support:

```bash
pip install pywin32
```

### 3. Optional Dependencies

For advanced features:

```bash
# For Whisper-based speech recognition (high accuracy, local)
pip install faster-whisper

# For wake word detection
pip install openwakeword
pip install pvporcupine  # Requires API key from Picovoice

# For transformer-based NLU
pip install transformers torch
```

### 4. Audio Setup

#### Windows
- PyAudio may require Microsoft Visual C++ 14.0 or greater
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Or install pre-built wheel: `pip install pipwin && pipwin install pyaudio`

#### Linux
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

## Quick Start

### 1. CLI Mode (Text-based)

The easiest way to test:

```bash
python voice_assistant_demo.py --mode cli
```

Type commands like:
- "what time is it?"
- "set a timer for 5 minutes"
- "hello"
- "help"

### 2. Demo Mode

Run sample commands:

```bash
python voice_assistant_demo.py --mode demo
```

### 3. Voice Mode

Voice-activated assistant:

```bash
python voice_assistant_demo.py --mode voice
```

Say the wake word (default: "assistant"), then speak your command.

### 4. Test Components

Test individual components:

```bash
python voice_assistant_demo.py --mode test
```

## Configuration

Edit `voice_assistant_config.yaml`:

```yaml
# Wake Word Detection
wake_word:
  words: ["assistant", "jarvis"]
  engine: "simple"  # or "porcupine", "openwakeword"
  sensitivity: 0.5

# Speech Recognition
speech:
  engine: "google"  # or "whisper", "windows"
  language: "en"

# TTS
tts:
  engine: "sapi"  # Windows SAPI for Windows 11
  rate: 150
  volume: 1.0
```

### Engine Options

#### Wake Word Detection
- `simple`: Energy-based (no dependencies, low accuracy)
- `openwakeword`: Open-source, runs locally (recommended)
- `porcupine`: High accuracy, requires API key

#### Speech Recognition
- `google`: Cloud-based, high accuracy (requires internet)
- `whisper`: Local, very high accuracy (requires faster-whisper)
- `windows`: Windows SAPI (native Windows support)

#### TTS
- `sapi`: Windows SAPI (best for Windows 11)
- `pyttsx3`: Cross-platform, offline
- `gtts`: Google TTS, high quality (requires internet)
- `edge`: Microsoft Edge TTS, high quality (requires internet)

## Usage Examples

### Python API

```python
from voice_assistant import VoiceAssistant

# Create assistant with default config
assistant = VoiceAssistant()

# Process text commands
response = assistant.process_text_command("what time is it?")
print(response)

# Run in CLI mode
assistant.run_cli()

# Run in voice mode
assistant.run()
```

### Creating Custom Plugins

```python
from voice_assistant.plugins.base_plugin import BasePlugin

class WeatherPlugin(BasePlugin):
    def get_intents(self):
        return ["weather"]

    def handle(self, intent_name, entities, context):
        location = entities.get("location", "your location")
        # Call weather API here
        return f"The weather in {location} is sunny"

# Register plugin
assistant.add_plugin(WeatherPlugin())
```

### Adding Custom Intents

```python
from voice_assistant import VoiceAssistant

config = {
    "nlu": {
        "backend": "pattern"
    }
}

assistant = VoiceAssistant(config)

# Add custom intent
assistant.intent_classifier.add_intent(
    name="movie",
    patterns=[r"\b(movie|film|cinema)\b"],
    examples=["play a movie", "find a film"],
    entities={"genre": r"\b(action|comedy|drama)\b"}
)
```

## Supported Commands

### Built-in Intents

- **Greeting**: "hello", "hi", "good morning"
- **Time**: "what time is it?", "current time"
- **Timer**: "set a timer for 5 minutes"
- **Reminder**: "remind me to call John"
- **Weather**: "what's the weather?", "weather in New York"
- **News**: "what's the news?", "tech news"
- **Search**: "search for Python tutorials"
- **Trading**: "stock price of AAPL", "buy 10 shares of TSLA"
- **Help**: "help", "what can you do?"
- **Goodbye**: "goodbye", "bye"

## Architecture

```
voice_assistant/
├── core/                 # Main orchestrator
│   └── assistant.py      # VoiceAssistant class
├── wake_word/            # Wake word detection
│   └── detector.py       # WakeWordDetector class
├── speech/               # Speech recognition
│   └── recognizer.py     # SpeechRecognizer class
├── nlu/                  # Natural Language Understanding
│   └── intent_classifier.py  # IntentClassifier class
├── tts/                  # Text-to-Speech
│   └── speech_synthesizer.py  # SpeechSynthesizer class
└── plugins/              # Plugin system
    ├── base_plugin.py    # BasePlugin & PluginManager
    └── builtin_plugins.py  # Built-in plugins
```

## Windows 11 Optimization

For best performance on Windows 11:

1. **Use Windows SAPI for TTS**:
   ```yaml
   tts:
     engine: "sapi"
   ```

2. **Configure Windows Speech Recognition**:
   - Go to Settings > Time & Language > Speech
   - Set up microphone and train speech recognition

3. **Install Windows-specific dependencies**:
   ```bash
   pip install pywin32
   ```

4. **Select appropriate voice**:
   ```python
   # List available voices
   assistant.tts.get_voices()

   # Set voice
   assistant.tts.set_voice("Microsoft David Desktop")
   ```

## Troubleshooting

### PyAudio Installation Issues
```bash
# Windows
pip install pipwin
pipwin install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Microphone Not Detected
- Check Windows Privacy Settings > Microphone
- Ensure apps have microphone access
- Test microphone in Sound Settings

### Wake Word Not Detecting
- Increase sensitivity in config
- Use `simple` engine for testing
- Check microphone input level

### Speech Recognition Fails
- Try different engine (google, whisper, windows)
- Check internet connection (for google/edge)
- Verify microphone is working

## Performance Tips

1. **Use local engines** for faster response:
   - Wake word: `openwakeword`
   - Speech: `whisper` (if GPU available)
   - TTS: `sapi` or `pyttsx3`

2. **Optimize Whisper model size**:
   - `tiny`: Fastest, lower accuracy
   - `base`: Balanced (recommended)
   - `small`: Better accuracy, slower
   - `medium/large`: Best accuracy, requires GPU

3. **Enable GPU acceleration** (if available):
   ```yaml
   speech:
     engine: "whisper"
     device: "cuda"
   ```

## Integration with Trading Bot

The voice assistant integrates with the trading bot:

```python
# Ask about stocks
"what's the stock price of AAPL?"
"show my portfolio"
"buy 10 shares of TSLA"
```

The TradingPlugin can be extended to:
- Execute trades
- Get market data
- Check portfolio status
- Set price alerts

## API Reference

### VoiceAssistant

```python
assistant = VoiceAssistant(config=None)
assistant.start()                    # Start voice mode
assistant.stop()                     # Stop assistant
assistant.run()                      # Run voice mode (blocking)
assistant.run_cli()                  # Run CLI mode
assistant.process_text_command(text) # Process text command
assistant.add_plugin(plugin)         # Add custom plugin
assistant.get_status()               # Get status dict
```

### Creating Plugins

```python
class MyPlugin(BasePlugin):
    def get_intents(self) -> List[str]:
        # Return list of intent names this plugin handles
        return ["my_intent"]

    def handle(self, intent_name: str, entities: Dict, context: Dict) -> str:
        # Process intent and return response text
        return "Response text"
```

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues.

## Support

For issues or questions:
1. Check Troubleshooting section
2. Review configuration
3. Enable debug mode: `--debug`
4. Check logs for errors
