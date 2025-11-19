# Quick Start Guide - AI Customer Service Bot

Get up and running in 5 minutes!

## Installation (One-Time Setup)

### Windows 11

1. **Install Python 3.9+**
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Open Command Prompt**
   - Press `Win + R`, type `cmd`, press Enter

3. **Navigate to Project**
   ```cmd
   cd path\to\ai-medical-diagnosis\customer_service_bot
   ```

4. **Install Dependencies**
   ```cmd
   pip install -r ..\requirements.txt
   python -m spacy download en_core_web_sm
   ```

   This will take 5-10 minutes (downloads AI models)

## Running the Bot

### Option 1: GUI Mode (Recommended for Windows 11)

**Double-click** `launch_bot.bat`

OR in Command Prompt:
```cmd
python bot_controller.py --gui
```

### Option 2: Command Line Mode

```cmd
python bot_controller.py --cli
```

Type your messages and press Enter. Type 'quit' to exit.

## First Conversation

1. **GUI Opens** - You'll see a modern chat interface

2. **Type a Message**
   ```
   How do I reset my password?
   ```

3. **Press Enter** - Bot responds with AI-generated answer

4. **Try Voice** (Optional)
   - Check "Enable Voice" in sidebar
   - Click "🎤 Voice" button
   - Speak your question
   - Bot responds with voice

5. **View Analytics**
   - Click "View Analytics" button
   - Browser opens with real-time dashboard

## Example Questions to Try

- "I need help with my account"
- "How can I track my order?"
- "What is your refund policy?"
- "I was charged twice for the same order"
- "When are you open?"
- "I want to speak to a manager" (triggers escalation)

## Features Demo

### Multi-Language
1. Select language from dropdown (Spanish, French, German, Japanese, etc.)
2. Type in that language
3. Bot detects and responds in same language

### Sentiment Analysis
- Type: "This is URGENT! I'm very frustrated!"
- Bot detects negative sentiment and high urgency
- May trigger escalation

### Knowledge Base Search
- Ask specific questions about policies
- Bot searches knowledge base for accurate answers
- Shows confidence scores

### Voice Features
1. Enable voice in sidebar
2. Click "🎤 Voice" for speech input
3. Bot speaks responses aloud

## Customization

### Edit Knowledge Base

Edit `customer_service_bot/data/knowledge_base.json`:

```json
{
  "question": "Your question here?",
  "answer": "Your answer here.",
  "category": "general_inquiry",
  "keywords": ["keyword1", "keyword2"]
}
```

Restart the bot to see changes.

### Change Settings

Edit `customer_service_bot/config.yaml`:

```yaml
# Change voice speed
voice:
  rate: 150  # words per minute (increase/decrease)

# Change AI model
models:
  dialogue:
    name: "microsoft/DialoGPT-large"  # for better quality
```

## Troubleshooting

### "Module not found" Error
```cmd
pip install -r ..\requirements.txt
```

### "spaCy model not found"
```cmd
python -m spacy download en_core_web_sm
```

### Voice Not Working
- Check microphone permissions in Windows Settings
- Install PyAudio: `pip install pyaudio`

### Slow Performance
- First message takes longer (loading models)
- Subsequent messages are faster
- Use GPU if available for best performance

### GUI Won't Open
- Check Python is installed: `python --version`
- Try CLI mode: `python bot_controller.py --cli`

## Next Steps

1. **Explore the GUI**
   - Try different features
   - Export conversations
   - View session statistics

2. **Customize Knowledge Base**
   - Add your own Q&A pairs
   - Categorize by intent
   - Import/export JSON

3. **View Analytics**
   - Real-time conversation metrics
   - Sentiment trends
   - Intent distribution

4. **Test Escalation**
   - Type urgent/angry messages
   - See automatic escalation

5. **Multi-Language**
   - Test different languages
   - See automatic translation

## Getting Help

- Read full README.md for detailed documentation
- Check module docstrings for API details
- Test individual components with test functions
- Email: support@company.com

## Performance Tips

- **First run is slow** - Models download on first use
- **Subsequent runs are faster** - Models cached locally
- **Use GPU** if available - Set `use_gpu: true` in config
- **Smaller models** - Use DialoGPT-small for faster responses

## What's Happening Under the Hood?

When you send a message:

1. **Language Detection** - Detects input language
2. **Translation** - Translates to English if needed
3. **Sentiment Analysis** - Analyzes emotion and urgency
4. **Intent Classification** - Determines what user wants
5. **Entity Extraction** - Finds names, dates, numbers, etc.
6. **Knowledge Base Search** - Searches for relevant answers
7. **Dialogue Generation** - Creates conversational response
8. **Escalation Check** - Determines if human needed
9. **Translation Back** - Translates response to user's language
10. **Voice Output** - Speaks response (if enabled)

All in under 2 seconds!

## Enjoy!

You now have a fully-functional AI customer service bot with enterprise-level features!

For advanced usage, see README.md
