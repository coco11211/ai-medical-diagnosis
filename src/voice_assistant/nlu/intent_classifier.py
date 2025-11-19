"""
Natural Language Understanding (NLU) Module

Provides intent classification and entity extraction using:
1. Rule-based pattern matching
2. Machine learning models (scikit-learn)
3. Optional integration with spaCy/transformers for advanced NLU
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Represents a detected intent with entities"""
    name: str
    confidence: float
    entities: Dict[str, Any]
    raw_text: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class IntentClassifier:
    """Intent classification with multiple backends"""

    def __init__(
        self,
        backend: str = "pattern",
        model_path: Optional[str] = None,
        intents_config: Optional[Dict] = None
    ):
        """
        Initialize intent classifier

        Args:
            backend: Classification backend ("pattern", "sklearn", "transformers")
            model_path: Path to trained model (for sklearn/transformers)
            intents_config: Intent configuration with patterns/examples
        """
        self.backend = backend
        self.model_path = model_path
        self.intents_config = intents_config or self._get_default_intents()

        self.model = None
        self.vectorizer = None
        self.label_encoder = None

        self._initialize_backend()

    def _get_default_intents(self) -> Dict:
        """Get default intent configurations"""
        return {
            "greeting": {
                "patterns": [
                    r"\b(hi|hello|hey|greetings)\b",
                    r"\bgood (morning|afternoon|evening)\b"
                ],
                "examples": [
                    "hello", "hi there", "hey", "good morning"
                ]
            },
            "goodbye": {
                "patterns": [
                    r"\b(bye|goodbye|see you|farewell)\b",
                    r"\btalk to you later\b"
                ],
                "examples": [
                    "goodbye", "bye", "see you later", "talk to you later"
                ]
            },
            "weather": {
                "patterns": [
                    r"\b(weather|temperature|forecast)\b",
                    r"\bhow('s| is) (the )?weather\b",
                    r"\bwill it (rain|snow)\b"
                ],
                "examples": [
                    "what's the weather", "how is the weather",
                    "weather forecast", "will it rain today"
                ],
                "entities": {
                    "location": r"\bin ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                    "date": r"\b(today|tomorrow|yesterday)\b"
                }
            },
            "time": {
                "patterns": [
                    r"\b(what time|what's the time|current time)\b",
                    r"\bwhat time is it\b"
                ],
                "examples": [
                    "what time is it", "current time", "what's the time"
                ]
            },
            "alarm": {
                "patterns": [
                    r"\bset (an? )?alarm\b",
                    r"\bwake me up\b"
                ],
                "examples": [
                    "set an alarm", "set alarm for 7 am",
                    "wake me up at 8"
                ],
                "entities": {
                    "time": r"\b(\d{1,2}:\d{2}|\d{1,2}\s?(?:am|pm))\b",
                    "duration": r"\bin (\d+) (minutes?|hours?)\b"
                }
            },
            "timer": {
                "patterns": [
                    r"\bset (a )?timer\b",
                    r"\bstart (a )?timer\b"
                ],
                "examples": [
                    "set a timer", "start timer for 5 minutes"
                ],
                "entities": {
                    "duration": r"\b(\d+)\s?(minutes?|hours?|seconds?)\b"
                }
            },
            "reminder": {
                "patterns": [
                    r"\bremind me\b",
                    r"\bset (a )?reminder\b"
                ],
                "examples": [
                    "remind me to call", "set a reminder"
                ],
                "entities": {
                    "task": r"remind me to (.+?)(?:\s+(?:at|in|on)|$)",
                    "time": r"\b(?:at |in |on )(.+)",
                }
            },
            "news": {
                "patterns": [
                    r"\b(news|headlines|what'?s happening)\b",
                    r"\btell me the news\b"
                ],
                "examples": [
                    "what's the news", "tell me the headlines",
                    "what's happening"
                ],
                "entities": {
                    "category": r"\b(tech|sports|business|entertainment|world)\s+news\b"
                }
            },
            "music": {
                "patterns": [
                    r"\bplay (music|song|artist)\b",
                    r"\b(stop|pause|resume|skip) (music|song)\b"
                ],
                "examples": [
                    "play music", "play some jazz", "pause music",
                    "skip song"
                ],
                "entities": {
                    "action": r"\b(play|stop|pause|resume|skip|next|previous)\b",
                    "genre": r"\b(rock|pop|jazz|classical|hip hop|electronic)\b",
                    "artist": r"\bplay (.+?) by (.+)",
                }
            },
            "search": {
                "patterns": [
                    r"\b(search|look up|find|google)\b",
                    r"\bwhat is|who is|where is\b"
                ],
                "examples": [
                    "search for python tutorial",
                    "who is Albert Einstein",
                    "what is machine learning"
                ],
                "entities": {
                    "query": r"(?:search|look up|find|google)\s+(?:for\s+)?(.+)|(?:what|who|where) is (.+)"
                }
            },
            "trading": {
                "patterns": [
                    r"\b(stock price|market|trading|portfolio)\b",
                    r"\bbuy|sell|trade\b"
                ],
                "examples": [
                    "what's the stock price of Apple",
                    "show my portfolio",
                    "buy 10 shares of Tesla"
                ],
                "entities": {
                    "symbol": r"\b([A-Z]{1,5})\b",
                    "action": r"\b(buy|sell|trade)\b",
                    "quantity": r"\b(\d+)\s+shares?\b"
                }
            },
            "help": {
                "patterns": [
                    r"\b(help|what can you do|commands|capabilities)\b"
                ],
                "examples": [
                    "help", "what can you do", "show commands"
                ]
            },
            "unknown": {
                "patterns": [],
                "examples": []
            }
        }

    def _initialize_backend(self):
        """Initialize the selected NLU backend"""
        if self.backend == "pattern":
            logger.info("Using pattern-based intent classification")
        elif self.backend == "sklearn":
            self._initialize_sklearn()
        elif self.backend == "transformers":
            self._initialize_transformers()
        else:
            logger.warning(f"Unknown backend '{self.backend}', using pattern-based")
            self.backend = "pattern"

    def _initialize_sklearn(self):
        """Initialize sklearn-based classifier"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.preprocessing import LabelEncoder

            if self.model_path and os.path.exists(self.model_path):
                # Load trained model
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                    self.label_encoder = data['label_encoder']
                logger.info(f"Loaded sklearn model from {self.model_path}")
            else:
                # Train new model from examples
                logger.info("Training new sklearn model from examples")
                self._train_sklearn_model()

        except ImportError:
            logger.error("scikit-learn not installed, falling back to pattern matching")
            self.backend = "pattern"

    def _train_sklearn_model(self):
        """Train sklearn model from intent examples"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder

        # Collect training data
        texts = []
        labels = []

        for intent_name, config in self.intents_config.items():
            examples = config.get('examples', [])
            for example in examples:
                texts.append(example.lower())
                labels.append(intent_name)

        if not texts:
            logger.warning("No training examples found, cannot train sklearn model")
            self.backend = "pattern"
            return

        # Train model
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.label_encoder = LabelEncoder()

        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        self.model = MultinomialNB()
        self.model.fit(X, y)

        logger.info(f"Trained sklearn model with {len(texts)} examples")

    def _initialize_transformers(self):
        """Initialize transformer-based classifier"""
        try:
            from transformers import pipeline

            if self.model_path:
                self.model = pipeline("text-classification", model=self.model_path)
            else:
                # Use zero-shot classification
                self.model = pipeline("zero-shot-classification")

            logger.info("Initialized transformer-based classifier")

        except ImportError:
            logger.error("transformers not installed, falling back to pattern matching")
            self.backend = "pattern"

    def classify(self, text: str) -> Intent:
        """
        Classify intent from text

        Args:
            text: Input text

        Returns:
            Intent object with name, confidence, and entities
        """
        text_lower = text.lower()

        if self.backend == "pattern":
            return self._classify_pattern(text, text_lower)
        elif self.backend == "sklearn":
            return self._classify_sklearn(text, text_lower)
        elif self.backend == "transformers":
            return self._classify_transformers(text, text_lower)
        else:
            return self._classify_pattern(text, text_lower)

    def _classify_pattern(self, text: str, text_lower: str) -> Intent:
        """Classify using pattern matching"""
        best_intent = "unknown"
        best_confidence = 0.0
        entities = {}

        for intent_name, config in self.intents_config.items():
            patterns = config.get('patterns', [])

            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    confidence = 0.8  # Base confidence for pattern match

                    # Extract entities if defined
                    entity_patterns = config.get('entities', {})
                    intent_entities = {}

                    for entity_name, entity_pattern in entity_patterns.items():
                        entity_match = re.search(entity_pattern, text, re.IGNORECASE)
                        if entity_match:
                            # Extract the first captured group or full match
                            intent_entities[entity_name] = (
                                entity_match.group(1) if entity_match.groups() else entity_match.group(0)
                            ).strip()

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent_name
                        entities = intent_entities

        return Intent(
            name=best_intent,
            confidence=best_confidence if best_confidence > 0 else 0.1,
            entities=entities,
            raw_text=text
        )

    def _classify_sklearn(self, text: str, text_lower: str) -> Intent:
        """Classify using sklearn model"""
        if not self.model:
            return self._classify_pattern(text, text_lower)

        X = self.vectorizer.transform([text_lower])
        proba = self.model.predict_proba(X)[0]
        best_idx = proba.argmax()
        confidence = float(proba[best_idx])
        intent_name = self.label_encoder.inverse_transform([best_idx])[0]

        # Extract entities using patterns
        entities = {}
        if intent_name in self.intents_config:
            entity_patterns = self.intents_config[intent_name].get('entities', {})
            for entity_name, entity_pattern in entity_patterns.items():
                entity_match = re.search(entity_pattern, text, re.IGNORECASE)
                if entity_match:
                    entities[entity_name] = (
                        entity_match.group(1) if entity_match.groups() else entity_match.group(0)
                    ).strip()

        return Intent(
            name=intent_name,
            confidence=confidence,
            entities=entities,
            raw_text=text
        )

    def _classify_transformers(self, text: str, text_lower: str) -> Intent:
        """Classify using transformer model"""
        if not self.model:
            return self._classify_pattern(text, text_lower)

        # Get intent labels
        candidate_labels = list(self.intents_config.keys())

        result = self.model(text, candidate_labels)
        intent_name = result['labels'][0]
        confidence = result['scores'][0]

        # Extract entities using patterns
        entities = {}
        if intent_name in self.intents_config:
            entity_patterns = self.intents_config[intent_name].get('entities', {})
            for entity_name, entity_pattern in entity_patterns.items():
                entity_match = re.search(entity_pattern, text, re.IGNORECASE)
                if entity_match:
                    entities[entity_name] = (
                        entity_match.group(1) if entity_match.groups() else entity_match.group(0)
                    ).strip()

        return Intent(
            name=intent_name,
            confidence=float(confidence),
            entities=entities,
            raw_text=text
        )

    def add_intent(self, name: str, patterns: List[str], examples: List[str] = None, entities: Dict[str, str] = None):
        """Add a new intent dynamically"""
        self.intents_config[name] = {
            "patterns": patterns,
            "examples": examples or [],
            "entities": entities or {}
        }
        logger.info(f"Added new intent: {name}")

        # Retrain sklearn model if using sklearn backend
        if self.backend == "sklearn":
            self._train_sklearn_model()

    def save_model(self, path: str):
        """Save trained model to disk"""
        if self.backend == "sklearn" and self.model:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'label_encoder': self.label_encoder
                }, f)
            logger.info(f"Saved sklearn model to {path}")
