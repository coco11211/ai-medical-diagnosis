"""
Intent Classification System
Identifies user intent from input text
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Dict, List, Tuple, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classifier for customer service interactions
    Identifies what the user wants to accomplish
    """

    # Predefined intents for customer service
    INTENTS = {
        'account_issue': {
            'description': 'Issues with account access or settings',
            'keywords': ['account', 'login', 'password', 'username', 'access', 'locked out', 'forgot'],
            'patterns': [
                r'can\'?t\s+log\s*in',
                r'forgot\s+(my\s+)?(password|username)',
                r'account\s+(is\s+)?(locked|blocked|suspended)',
                r'reset\s+(my\s+)?password'
            ]
        },
        'billing_payment': {
            'description': 'Billing, payment, or subscription questions',
            'keywords': ['bill', 'payment', 'charge', 'subscription', 'invoice', 'refund', 'price', 'cost'],
            'patterns': [
                r'charged\s+me',
                r'billing\s+issue',
                r'cancel\s+(my\s+)?subscription',
                r'refund'
            ]
        },
        'technical_support': {
            'description': 'Technical problems or bugs',
            'keywords': ['error', 'bug', 'not working', 'broken', 'crash', 'issue', 'problem', 'fix'],
            'patterns': [
                r'(not|isn\'t|doesn\'t)\s+work(ing)?',
                r'getting\s+(an\s+)?error',
                r'keeps\s+crash(ing)?',
                r'technical\s+(issue|problem)'
            ]
        },
        'product_inquiry': {
            'description': 'Questions about products or features',
            'keywords': ['how to', 'what is', 'feature', 'product', 'explain', 'tell me about'],
            'patterns': [
                r'how\s+(do|can)\s+i',
                r'what\s+(is|are|does)',
                r'tell\s+me\s+about',
                r'explain'
            ]
        },
        'order_status': {
            'description': 'Order tracking and shipping',
            'keywords': ['order', 'shipping', 'delivery', 'tracking', 'package', 'shipment', 'arrived'],
            'patterns': [
                r'track\s+(my\s+)?order',
                r'where\s+(is\s+)?(my\s+)?(order|package)',
                r'shipping\s+status',
                r'when\s+will\s+.+\s+arrive'
            ]
        },
        'complaint': {
            'description': 'Customer complaints',
            'keywords': ['complaint', 'disappointed', 'unsatisfied', 'terrible', 'awful', 'unacceptable'],
            'patterns': [
                r'file\s+(a\s+)?complaint',
                r'speak\s+to\s+(a\s+)?manager',
                r'this\s+is\s+(un)?acceptable',
                r'very\s+(un)?(satisfied|happy)'
            ]
        },
        'return_exchange': {
            'description': 'Product returns or exchanges',
            'keywords': ['return', 'exchange', 'refund', 'send back', 'replace'],
            'patterns': [
                r'return\s+(this|the|my)',
                r'exchange\s+for',
                r'send\s+(it\s+)?back',
                r'get\s+a\s+refund'
            ]
        },
        'general_inquiry': {
            'description': 'General questions',
            'keywords': ['question', 'ask', 'wonder', 'curious', 'information'],
            'patterns': [
                r'i\s+have\s+a\s+question',
                r'can\s+you\s+help',
                r'i\s+was\s+wondering'
            ]
        },
        'greeting': {
            'description': 'Greeting or conversation starter',
            'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'greetings'],
            'patterns': [
                r'^(hi|hello|hey)[\s\!]',
                r'good\s+(morning|afternoon|evening)',
            ]
        },
        'farewell': {
            'description': 'Ending conversation',
            'keywords': ['bye', 'goodbye', 'thanks', 'thank you', 'that\'s all'],
            'patterns': [
                r'(good)?bye',
                r'thank\s+(you|u)',
                r'that\'?s\s+all',
                r'have\s+a\s+(good|nice)\s+day'
            ]
        }
    }

    def __init__(
        self,
        use_transformer: bool = True,
        transformer_model: str = "facebook/bart-large-mnli",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize intent classifier

        Args:
            use_transformer: Whether to use transformer for zero-shot classification
            transformer_model: Model for zero-shot classification
            confidence_threshold: Minimum confidence for intent prediction
        """
        self.confidence_threshold = confidence_threshold
        self.use_transformer = use_transformer

        if use_transformer:
            try:
                logger.info(f"Loading intent classifier model: {transformer_model}")
                device = 0 if torch.cuda.is_available() else -1

                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=transformer_model,
                    device=device
                )

                # Prepare candidate labels
                self.candidate_labels = [
                    intent_data['description']
                    for intent_data in self.INTENTS.values()
                ]

                logger.info("Intent classifier initialized successfully")

            except Exception as e:
                logger.error(f"Error loading transformer model: {e}")
                logger.info("Falling back to rule-based classification")
                self.use_transformer = False

    def classify(self, text: str, top_k: int = 3) -> Dict:
        """
        Classify user intent from text

        Args:
            text: Input text
            top_k: Number of top intents to return

        Returns:
            Dictionary with intent predictions
        """
        if not text or not text.strip():
            return self._empty_result()

        text = text.strip()

        # Rule-based classification (always run for fallback)
        rule_based_scores = self._rule_based_classification(text)

        # Transformer-based classification
        if self.use_transformer:
            try:
                transformer_scores = self._transformer_classification(text)

                # Ensemble: combine rule-based and transformer scores
                final_scores = self._ensemble_scores(rule_based_scores, transformer_scores)

            except Exception as e:
                logger.error(f"Error in transformer classification: {e}")
                final_scores = rule_based_scores
        else:
            final_scores = rule_based_scores

        # Get top-k intents
        sorted_intents = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Format results
        top_intent = sorted_intents[0][0] if sorted_intents else 'general_inquiry'
        confidence = sorted_intents[0][1] if sorted_intents else 0.0

        result = {
            'intent': top_intent,
            'confidence': round(confidence, 4),
            'all_intents': [
                {
                    'intent': intent,
                    'confidence': round(score, 4),
                    'description': self.INTENTS[intent]['description']
                }
                for intent, score in sorted_intents
            ],
            'requires_human': confidence < self.confidence_threshold or top_intent == 'complaint'
        }

        logger.debug(f"Classified intent: {top_intent} (confidence: {confidence:.3f})")

        return result

    def _rule_based_classification(self, text: str) -> Dict[str, float]:
        """
        Rule-based intent classification using keywords and patterns

        Returns:
            Dictionary mapping intent names to confidence scores
        """
        text_lower = text.lower()
        scores = {intent: 0.0 for intent in self.INTENTS.keys()}

        for intent, data in self.INTENTS.items():
            score = 0.0

            # Keyword matching
            keyword_matches = sum(
                1 for keyword in data['keywords']
                if keyword in text_lower
            )
            score += keyword_matches * 0.3

            # Pattern matching
            pattern_matches = sum(
                1 for pattern in data['patterns']
                if re.search(pattern, text_lower, re.IGNORECASE)
            )
            score += pattern_matches * 0.5

            scores[intent] = min(score, 1.0)

        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _transformer_classification(self, text: str) -> Dict[str, float]:
        """
        Transformer-based zero-shot classification

        Returns:
            Dictionary mapping intent names to confidence scores
        """
        result = self.classifier(
            text,
            self.candidate_labels,
            multi_label=True
        )

        # Map labels back to intent names
        scores = {}
        intent_list = list(self.INTENTS.keys())

        for label, score in zip(result['labels'], result['scores']):
            # Find matching intent by description
            for intent, data in self.INTENTS.items():
                if data['description'] == label:
                    scores[intent] = score
                    break

        return scores

    def _ensemble_scores(
        self,
        rule_scores: Dict[str, float],
        transformer_scores: Dict[str, float],
        rule_weight: float = 0.3,
        transformer_weight: float = 0.7
    ) -> Dict[str, float]:
        """
        Combine rule-based and transformer scores

        Args:
            rule_scores: Scores from rule-based classifier
            transformer_scores: Scores from transformer classifier
            rule_weight: Weight for rule-based scores
            transformer_weight: Weight for transformer scores

        Returns:
            Combined scores
        """
        ensemble = {}

        for intent in self.INTENTS.keys():
            rule_score = rule_scores.get(intent, 0.0)
            trans_score = transformer_scores.get(intent, 0.0)

            ensemble[intent] = (
                rule_weight * rule_score +
                transformer_weight * trans_score
            )

        return ensemble

    def _empty_result(self) -> Dict:
        """Return empty result"""
        return {
            'intent': 'general_inquiry',
            'confidence': 0.0,
            'all_intents': [],
            'requires_human': True
        }

    def get_intent_description(self, intent: str) -> str:
        """Get description for an intent"""
        return self.INTENTS.get(intent, {}).get('description', '')

    def add_custom_intent(
        self,
        intent_name: str,
        description: str,
        keywords: List[str],
        patterns: List[str]
    ):
        """
        Add a custom intent to the classifier

        Args:
            intent_name: Name of the intent
            description: Description for zero-shot classification
            keywords: List of keywords
            patterns: List of regex patterns
        """
        self.INTENTS[intent_name] = {
            'description': description,
            'keywords': keywords,
            'patterns': patterns
        }

        # Update candidate labels for transformer
        if self.use_transformer:
            self.candidate_labels = [
                intent_data['description']
                for intent_data in self.INTENTS.values()
            ]

        logger.info(f"Added custom intent: {intent_name}")

    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts"""
        return [self.classify(text) for text in texts]


# Testing function
def test_intent_classifier():
    """Test intent classifier"""
    classifier = IntentClassifier()

    test_texts = [
        "I can't log in to my account",
        "I was charged twice for the same order",
        "How do I track my package?",
        "This product is terrible! I want a refund!",
        "What features does your premium plan have?",
        "Hi, I need some help",
        "Thank you for your help!",
        "The app keeps crashing when I try to upload photos"
    ]

    print("Intent Classification Test")
    print("=" * 70)

    for text in test_texts:
        print(f"\nText: {text}")
        result = classifier.classify(text, top_k=3)

        print(f"Primary Intent: {result['intent']} "
              f"(confidence: {result['confidence']:.3f})")
        print("Top 3 Intents:")
        for intent_data in result['all_intents']:
            print(f"  - {intent_data['intent']}: {intent_data['confidence']:.3f} "
                  f"({intent_data['description']})")
        print(f"Requires Human: {result['requires_human']}")
        print("-" * 70)


if __name__ == "__main__":
    test_intent_classifier()
