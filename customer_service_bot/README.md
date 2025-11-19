# AI Customer Service Bot for Windows 11

A state-of-the-art conversational AI customer service bot built with transformer-based models, featuring multi-turn context management, sentiment analysis, intent classification, entity extraction, knowledge base integration, smart escalation logic, multi-language support, voice integration, and a real-time analytics dashboard.

## Features

### Core AI Capabilities

- **Transformer-Based Dialogue System**
  - Uses Microsoft DialoGPT for natural conversations
  - Multi-turn context management (maintains conversation history)
  - Context-aware response generation
  - Session persistence and export

- **Sentiment Analysis**
  - Multi-model ensemble approach (BERT, VADER, TextBlob)
  - Real-time emotion detection
  - Sentiment trend analysis
  - Urgency scoring for prioritization

- **Intent Classification**
  - Zero-shot classification with BART
  - Rule-based classification for accuracy
  - Handles 10+ predefined intents
  - Custom intent support
  - Confidence scoring

- **Entity Extraction**
  - Named Entity Recognition (NER) with BERT
  - Custom pattern matching for domain-specific entities
  - Extracts: emails, phone numbers, order numbers, dates, currencies, etc.
  - Multi-source entity consolidation

### Advanced Features

- **Knowledge Base with Vector Search**
  - Semantic search using sentence transformers
  - FAISS vector database for fast retrieval
  - Supports JSON import/export
  - Category-based filtering
  - Relevance scoring

- **Smart Escalation Logic**
  - Rule-based escalation triggers
  - Sentiment-driven escalation
  - Urgency-based prioritization
  - Automatic routing to appropriate departments
  - Escalation analytics

- **Multi-Language Support**
  - Supports 30+ languages
  - Automatic language detection
  - Real-time translation
  - Language-specific formatting
  - Maintains context across translations

- **Voice Integration**
  - Speech recognition (Google Speech API + Sphinx offline)
  - Text-to-speech with customizable voices
  - Background listening mode
  - Multi-microphone support
  - Voice activity detection

- **Analytics Dashboard**
  - Real-time conversation metrics
  - Sentiment trends visualization
  - Intent distribution charts
  - Language usage statistics
  - Escalation tracking
  - Resolution rate monitoring

- **Modern Windows 11 GUI**
  - Clean, modern interface
  - Real-time chat display
  - Session management
  - Voice controls
  - Multi-language selector
  - Export functionality

## System Requirements

- **Operating System**: Windows 11 (compatible with Windows 10, Linux, macOS)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **Storage**: 5GB free space (for models and data)
- **GPU**: Optional but recommended for faster inference
- **Microphone**: Required for voice features

## Installation

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd ai-medical-diagnosis/customer_service_bot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r ../requirements.txt
```

**Note**: First installation may take 10-15 minutes as it downloads transformer models.

### Step 4: Download SpaCy Model (Required)

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Configure (Optional)

Edit `customer_service_bot/config.yaml` to customize settings.

## Quick Start

### GUI Mode (Default)

```bash
python bot_controller.py
```

Or explicitly:

```bash
python bot_controller.py --gui
```

### CLI Mode

```bash
python bot_controller.py --cli
```

## Usage Guide

### Basic Conversation

1. **Start the Application**
   - Run `python bot_controller.py`
   - GUI window opens automatically

2. **Begin Chatting**
   - Type your message in the input box
   - Press Enter or click "Send"
   - Bot responds with AI-generated answer

3. **View Analytics**
   - Click "View Analytics" in sidebar
   - Dashboard opens in browser at http://127.0.0.1:8051

### Using Voice Features

1. **Enable Voice**
   - Check "Enable Voice" in the sidebar
   - Bot will speak responses aloud

2. **Voice Input**
   - Click the "🎤 Voice" button
   - Speak when prompted
   - Text appears in input box
   - Click "Send" to send message

### Multi-Language Support

1. **Select Language**
   - Choose language from dropdown (en, es, fr, de, etc.)
   - Type or speak in selected language
   - Bot detects language and responds appropriately

2. **Automatic Detection**
   - Bot automatically detects input language
   - Translates to English for processing
   - Responds in user's language

### Managing Conversations

- **New Conversation**: Click "New Conversation" button
- **Export**: Click "Export Conversation" to save as JSON
- **Session Info**: View message count and duration in sidebar

## Project Structure

```
customer_service_bot/
├── dialogue/
│   └── dialogue_manager.py       # Transformer-based dialogue system
├── nlp/
│   ├── sentiment_analyzer.py     # Sentiment analysis
│   ├── intent_classifier.py      # Intent classification
│   ├── entity_extractor.py       # Entity extraction
│   └── language_support.py       # Multi-language support
├── knowledge_base/
│   └── knowledge_base.py          # Vector search knowledge base
├── utils/
│   └── escalation.py              # Escalation logic
├── voice/
│   └── voice_interface.py         # Voice I/O
├── analytics/
│   └── analytics_dashboard.py     # Analytics & dashboard
├── gui/
│   └── main_window.py             # GUI application
├── data/
│   └── knowledge_base.json        # Knowledge base entries
├── config.yaml                    # Configuration file
├── bot_controller.py              # Main controller
└── README.md                      # This file
```

## Configuration

### Edit `config.yaml` to customize:

**NLP Models**:
```yaml
models:
  dialogue:
    name: "microsoft/DialoGPT-medium"  # or DialoGPT-large
    temperature: 0.7
    max_context_turns: 5
```

**Escalation Settings**:
```yaml
escalation:
  sentiment_threshold: -0.5
  urgency_threshold: 7.0
  unresolved_turns_threshold: 3
```

**Voice Settings**:
```yaml
voice:
  language: "en-US"
  rate: 150  # words per minute
  volume: 0.9
  gender: "female"
```

## Knowledge Base Management

### Add Entries Programmatically

```python
from knowledge_base.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

kb.add_entry(
    question="How do I reset my password?",
    answer="Click 'Forgot Password' and follow the instructions.",
    category="account_issue",
    keywords=["password", "reset", "login"]
)

# Save
kb.export_to_json("data/knowledge_base.json")
```

### Import from JSON

```python
kb.import_from_json("data/knowledge_base.json")
```

## Testing Individual Components

### Test Sentiment Analysis

```bash
python customer_service_bot/nlp/sentiment_analyzer.py
```

### Test Intent Classification

```bash
python customer_service_bot/nlp/intent_classifier.py
```

### Test Entity Extraction

```bash
python customer_service_bot/nlp/entity_extractor.py
```

### Test Knowledge Base

```bash
python customer_service_bot/knowledge_base/knowledge_base.py
```

### Test Voice Interface

```bash
python customer_service_bot/voice/voice_interface.py
```

### Test Analytics Dashboard

```bash
python customer_service_bot/analytics/analytics_dashboard.py
```

## Analytics Dashboard

### Access Dashboard

1. Start the application
2. Click "View Analytics" button
3. Browser opens to http://127.0.0.1:8051

### Metrics Displayed

- **Total Conversations**: Number of chat sessions
- **Total Messages**: Total messages exchanged
- **Escalations**: Number of escalated conversations
- **Average Sentiment**: Overall sentiment score
- **Conversation Timeline**: 24-hour conversation volume
- **Sentiment Timeline**: Sentiment trends over time
- **Intent Distribution**: Pie chart of detected intents
- **Language Distribution**: Languages used by customers

## API Reference

### CustomerServiceBot

```python
from bot_controller import CustomerServiceBot

# Initialize
bot = CustomerServiceBot(
    use_voice=False,
    language='en',
    knowledge_base_path='data/knowledge_base.json'
)

# Start session
bot.start_session(session_id='session_001')

# Process message
response = bot.process_message(
    user_input="I need help with my account",
    session_id='session_001',
    language='en'
)

# Response structure
{
    'response': 'How can I help you with your account?',
    'sentiment': {...},
    'intent': {...},
    'entities': {...},
    'escalation': {...},
    'session_info': {...}
}

# End session
bot.end_session(session_id='session_001')
```

## Troubleshooting

### Issue: Models Not Downloading

**Solution**:
```bash
# Manually install transformers
pip install --upgrade transformers torch

# Test download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-medium')"
```

### Issue: Voice Not Working

**Solution**:
- Install PyAudio: `pip install pyaudio`
- On Windows, may need to install from wheel:
  Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Check microphone permissions in Windows Settings

### Issue: SpaCy Model Not Found

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Issue: GPU Not Detected

**Solution**:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Dashboard Not Opening

**Solution**:
- Check port 8051 is not in use
- Change port in config.yaml
- Manually open: http://127.0.0.1:8051

## Performance Optimization

### GPU Acceleration

Models automatically use GPU if available. To force CPU:

```python
bot = CustomerServiceBot()
bot.dialogue_manager.device = 'cpu'
```

### Model Size Selection

For faster performance with less RAM:

```yaml
models:
  dialogue:
    name: "microsoft/DialoGPT-small"  # Instead of medium/large
```

### Reduce Context Window

```yaml
models:
  dialogue:
    max_context_turns: 3  # Instead of 5
```

## Security Considerations

- **Data Privacy**: Conversations are stored locally only
- **API Keys**: Use environment variables for any external API keys
- **Session Security**: Sessions timeout after 30 minutes of inactivity
- **Input Validation**: All user inputs are sanitized
- **Knowledge Base**: Review entries for sensitive information before deployment

## Extending the Bot

### Add Custom Intents

```python
from nlp.intent_classifier import IntentClassifier

classifier = IntentClassifier()

classifier.add_custom_intent(
    intent_name='warranty_inquiry',
    description='Questions about product warranty',
    keywords=['warranty', 'guarantee', 'coverage'],
    patterns=[r'warranty.*product', r'guarantee.*cover']
)
```

### Add Custom Escalation Rules

```python
from utils.escalation import EscalationManager

escalation = EscalationManager()

escalation.escalation_rules.append({
    'id': 'custom_rule',
    'description': 'Custom escalation condition',
    'check': lambda ctx: ctx.get('custom_flag', False),
    'priority': 'high',
    'reason': 'Custom reason'
})
```

## Deployment

### Windows Service

Create a Windows service using `pywin32` or `NSSM` for production deployment.

### Docker Deployment

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY customer_service_bot/ ./customer_service_bot/
CMD ["python", "customer_service_bot/bot_controller.py", "--cli"]
```

## License

This project is for educational and commercial use.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: support@company.com
- Check the documentation in each module

## Acknowledgments

Built with:
- Transformers (Hugging Face)
- PyTorch
- spaCy
- FAISS
- Dash/Plotly
- SpeechRecognition
- pyttsx3

## Changelog

### Version 1.0.0
- Initial release
- Transformer-based dialogue
- Multi-language support
- Voice integration
- Analytics dashboard
- Smart escalation
- Knowledge base with vector search
