"""
Main Bot Controller
Integrates all components of the customer service bot
"""

from dialogue.dialogue_manager import DialogueManager, ConversationContext
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.intent_classifier import IntentClassifier
from nlp.entity_extractor import EntityExtractor
from nlp.language_support import LanguageManager
from knowledge_base.knowledge_base import KnowledgeBase
from utils.escalation import EscalationManager
from voice.voice_interface import VoiceInterface
from analytics.analytics_dashboard import ConversationAnalytics, AnalyticsDashboard

import logging
from typing import Dict, Optional
from datetime import datetime
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerServiceBot:
    """
    Main controller for customer service bot
    Integrates all NLP and AI components
    """

    def __init__(
        self,
        use_voice: bool = False,
        language: str = 'en',
        knowledge_base_path: Optional[str] = None
    ):
        """
        Initialize customer service bot

        Args:
            use_voice: Enable voice interface
            language: Default language
            knowledge_base_path: Path to knowledge base JSON file
        """
        logger.info("Initializing Customer Service Bot...")

        # Initialize NLP components
        logger.info("Loading NLP components...")
        self.dialogue_manager = DialogueManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.language_manager = LanguageManager(default_language=language)

        # Initialize knowledge base
        logger.info("Loading knowledge base...")
        self.knowledge_base = KnowledgeBase()
        if knowledge_base_path:
            self.knowledge_base.import_from_json(knowledge_base_path)
        else:
            self._load_default_knowledge_base()

        # Initialize escalation manager
        self.escalation_manager = EscalationManager()

        # Initialize voice interface (optional)
        self.voice_interface = None
        if use_voice:
            try:
                self.voice_interface = VoiceInterface(language=f'{language}-US')
                logger.info("Voice interface enabled")
            except Exception as e:
                logger.warning(f"Voice interface not available: {e}")

        # Initialize analytics
        self.analytics = ConversationAnalytics()
        self.analytics_dashboard = None

        # Active sessions
        self.active_sessions: Dict[str, Dict] = {}

        logger.info("Customer Service Bot initialized successfully!")

    def _load_default_knowledge_base(self):
        """Load default knowledge base entries"""
        default_entries = [
            {
                'question': 'How do I reset my password?',
                'answer': 'To reset your password: 1) Click "Forgot Password" on the login page, 2) Enter your email address, 3) Check your email for a reset link, 4) Follow the link and create a new password.',
                'category': 'account_issue',
                'keywords': ['password', 'reset', 'login', 'forgot']
            },
            {
                'question': 'What are your business hours?',
                'answer': 'Our customer service is available Monday-Friday 9AM-6PM EST, Saturday 10AM-4PM EST, and Sunday 12PM-4PM EST. For urgent matters outside these hours, please email support@company.com.',
                'category': 'general_inquiry',
                'keywords': ['hours', 'time', 'when', 'available']
            },
            {
                'question': 'How can I track my order?',
                'answer': 'To track your order: 1) Log into your account, 2) Go to "Order History", 3) Click on your order number to see tracking details. You can also use the tracking number sent to your email on the carrier website.',
                'category': 'order_status',
                'keywords': ['track', 'order', 'shipping', 'delivery']
            },
            {
                'question': 'What is your refund policy?',
                'answer': 'We offer full refunds within 30 days of purchase. Items must be unused and in original packaging. To request a refund, contact us with your order number. Refunds are processed within 5-7 business days.',
                'category': 'return_exchange',
                'keywords': ['refund', 'return', 'money back', 'policy']
            },
            {
                'question': 'How do I cancel my subscription?',
                'answer': 'To cancel your subscription: 1) Log into your account, 2) Go to Settings > Billing, 3) Click "Cancel Subscription", 4) Confirm cancellation. Your subscription will remain active until the end of the current billing period.',
                'category': 'billing_payment',
                'keywords': ['cancel', 'subscription', 'billing', 'stop']
            },
            {
                'question': 'I\'m having technical issues',
                'answer': 'For technical issues: 1) Try clearing your browser cache and cookies, 2) Update to the latest browser version, 3) Disable browser extensions, 4) Try a different browser. If issues persist, contact our technical support team.',
                'category': 'technical_support',
                'keywords': ['technical', 'issue', 'problem', 'not working', 'error']
            }
        ]

        self.knowledge_base.add_entries_batch(default_entries)
        logger.info(f"Loaded {len(default_entries)} default knowledge base entries")

    def start_session(self, session_id: str) -> Dict:
        """
        Start a new conversation session

        Args:
            session_id: Unique session identifier

        Returns:
            Session information
        """
        # Create dialogue context
        context = self.dialogue_manager.create_session(session_id)

        # Initialize session tracking
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'start_time': datetime.now(),
            'message_count': 0,
            'language': self.language_manager.default_language,
            'escalated': False,
            'resolved': False,
            'sentiment_history': [],
            'intents': []
        }

        logger.info(f"Started session: {session_id}")

        return {
            'session_id': session_id,
            'status': 'active',
            'greeting': self.language_manager.format_greeting(
                self.active_sessions[session_id]['language']
            )
        }

    def process_message(
        self,
        user_input: str,
        session_id: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Process user message and generate response

        Args:
            user_input: User's input message
            session_id: Session identifier
            language: Override language for this message

        Returns:
            Response dictionary with all analysis results
        """
        # Get or create session
        if session_id not in self.active_sessions:
            self.start_session(session_id)

        session = self.active_sessions[session_id]
        session['message_count'] += 1

        # Language processing
        if language:
            session['language'] = language

        lang_result = self.language_manager.process_user_input(
            user_input,
            session['language']
        )

        # Use English text for processing
        processing_text = lang_result['english_text']

        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze(processing_text)
        session['sentiment_history'].append(sentiment_result['ensemble']['score'])

        # Intent classification
        intent_result = self.intent_classifier.classify(processing_text)
        session['intents'].append(intent_result['intent'])

        # Entity extraction
        entity_result = self.entity_extractor.extract(processing_text)

        # Search knowledge base
        kb_results = self.knowledge_base.search(
            processing_text,
            k=3,
            category_filter=intent_result['intent'] if intent_result['confidence'] > 0.6 else None
        )

        # Check for escalation
        escalation_context = {
            'text': processing_text,
            'sentiment': sentiment_result['ensemble'],
            'urgency': sentiment_result['urgency'],
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'unresolved_turns': self._count_unresolved_turns(session),
            'sentiment_trend': self._get_sentiment_trend(session),
            'entities': entity_result['consolidated_entities']
        }

        should_escalate, escalation_info = self.escalation_manager.should_escalate(
            escalation_context
        )

        # Generate response
        if should_escalate:
            # Escalation response
            routing_info = self.escalation_manager.get_routing_info(
                escalation_info,
                escalation_context
            )
            english_response = self.escalation_manager.get_escalation_message(routing_info)
            session['escalated'] = True

        elif kb_results and kb_results[0]['score'] > 0.7:
            # Use knowledge base answer
            english_response = kb_results[0]['answer']
            session['resolved'] = True

        else:
            # Generate dialogue response
            dialogue_response, metadata = self.dialogue_manager.generate_response(
                processing_text,
                session_id
            )
            english_response = dialogue_response

        # Translate response if needed
        if session['language'] != 'en':
            response_translation = self.language_manager.create_multilingual_response(
                english_response,
                session['language']
            )
            final_response = response_translation['user_language']
        else:
            final_response = english_response

        # Speak response if voice enabled
        if self.voice_interface:
            threading.Thread(
                target=self.voice_interface.speak,
                args=(final_response,),
                daemon=True
            ).start()

        # Build complete response
        response = {
            'response': final_response,
            'english_response': english_response,
            'sentiment': sentiment_result,
            'intent': intent_result,
            'entities': entity_result,
            'language': lang_result,
            'knowledge_base_results': kb_results,
            'escalation': {
                'should_escalate': should_escalate,
                'info': escalation_info if should_escalate else {},
                'message': english_response if should_escalate else None
            },
            'session_info': {
                'message_count': session['message_count'],
                'escalated': session['escalated']
            }
        }

        return response

    def end_session(self, session_id: str) -> Dict:
        """End a conversation session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}

        session = self.active_sessions[session_id]

        # Calculate session metrics
        duration = (datetime.now() - session['start_time']).seconds
        avg_sentiment = sum(session['sentiment_history']) / max(len(session['sentiment_history']), 1)

        # Log to analytics
        self.analytics.log_conversation({
            'session_id': session_id,
            'message_count': session['message_count'],
            'avg_sentiment': avg_sentiment,
            'primary_intent': max(set(session['intents']), key=session['intents'].count) if session['intents'] else 'unknown',
            'language': session['language'],
            'escalated': session['escalated'],
            'resolved': session['resolved'],
            'duration_seconds': duration
        })

        # End dialogue session
        summary = self.dialogue_manager.end_session(session_id)

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(f"Ended session: {session_id}")

        return summary

    def _count_unresolved_turns(self, session: Dict) -> int:
        """Count unresolved conversation turns"""
        # Simple heuristic: if sentiment is declining, consider unresolved
        if len(session['sentiment_history']) < 2:
            return 0

        recent_sentiments = session['sentiment_history'][-3:]
        if all(recent_sentiments[i] <= recent_sentiments[i-1] for i in range(1, len(recent_sentiments))):
            return len(recent_sentiments)

        return 0

    def _get_sentiment_trend(self, session: Dict) -> str:
        """Get sentiment trend for session"""
        if len(session['sentiment_history']) < 2:
            return 'stable'

        trend_result = self.sentiment_analyzer.get_sentiment_trend(
            session['sentiment_history']
        )
        return trend_result['trend']

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        if session_id not in self.active_sessions:
            return {}

        session = self.active_sessions[session_id]
        duration = (datetime.now() - session['start_time']).seconds

        return {
            'session_id': session_id,
            'message_count': session['message_count'],
            'duration': duration,
            'language': session['language'],
            'escalated': session['escalated'],
            'avg_sentiment': sum(session['sentiment_history']) / max(len(session['sentiment_history']), 1)
        }

    def set_voice_enabled(self, enabled: bool):
        """Enable or disable voice"""
        if enabled and not self.voice_interface:
            try:
                self.voice_interface = VoiceInterface()
            except Exception as e:
                logger.error(f"Could not enable voice: {e}")
        elif not enabled:
            self.voice_interface = None

    def get_voice_input(self) -> Optional[str]:
        """Get voice input"""
        if not self.voice_interface:
            return None

        result = self.voice_interface.listen(timeout=5, phrase_time_limit=10)
        return result.get('text', '') if result.get('success') else None

    def open_analytics_dashboard(self):
        """Open analytics dashboard in browser"""
        if not self.analytics_dashboard:
            self.analytics_dashboard = AnalyticsDashboard(self.analytics, port=8051)

        # Run in thread
        threading.Thread(
            target=self.analytics_dashboard.run,
            daemon=True
        ).start()

    def export_session(self, session_id: str, filepath: str):
        """Export session to file"""
        self.dialogue_manager.export_session(session_id, filepath)


# Main entry point
def main():
    """Main entry point for the application"""
    import sys

    # Initialize bot
    bot = CustomerServiceBot(use_voice=False)

    # Check if GUI mode
    if '--gui' in sys.argv or len(sys.argv) == 1:
        # Run GUI
        from gui.main_window import CustomerServiceBotGUI
        app = CustomerServiceBotGUI(bot)
        app.run()

    elif '--cli' in sys.argv:
        # Run CLI mode
        print("AI Customer Service Bot - CLI Mode")
        print("=" * 50)

        session_id = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bot.start_session(session_id)

        print("\nType 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    bot.end_session(session_id)
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                response = bot.process_message(user_input, session_id)
                print(f"Bot: {response['response']}\n")

            except KeyboardInterrupt:
                bot.end_session(session_id)
                print("\nGoodbye!")
                break

    else:
        print("Usage:")
        print("  python bot_controller.py --gui    # Run GUI mode (default)")
        print("  python bot_controller.py --cli    # Run CLI mode")


if __name__ == "__main__":
    main()
