"""
Sentiment Analysis Module
Analyzes sentiment of user messages for escalation and analytics
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
from typing import Dict, Tuple, List
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer using:
    - Transformer-based model (BERT/RoBERTa)
    - VADER (Valence Aware Dictionary and sEntiment Reasoner)
    - TextBlob for polarity and subjectivity
    """

    def __init__(
        self,
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        use_gpu: bool = True
    ):
        """
        Initialize sentiment analyzer

        Args:
            transformer_model: HuggingFace model for sentiment analysis
            use_gpu: Whether to use GPU if available
        """
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1

        logger.info(f"Initializing sentiment analyzer with model: {transformer_model}")

        try:
            # Initialize transformer-based sentiment analyzer
            self.transformer_sentiment = pipeline(
                "sentiment-analysis",
                model=transformer_model,
                device=self.device
            )

            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()

            logger.info("Sentiment analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            raise

        # Sentiment thresholds for classification
        self.thresholds = {
            'very_negative': -0.7,
            'negative': -0.3,
            'neutral': 0.3,
            'positive': 0.7
        }

    def analyze(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment scores and classification
        """
        if not text or not text.strip():
            return self._empty_result()

        results = {}

        try:
            # Transformer-based sentiment
            transformer_result = self.transformer_sentiment(text[:512])[0]
            results['transformer'] = {
                'label': transformer_result['label'],
                'score': transformer_result['score']
            }

            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            results['vader'] = vader_scores

            # TextBlob sentiment
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

            # Ensemble sentiment (weighted average)
            ensemble_score = self._calculate_ensemble(results)
            results['ensemble'] = {
                'score': ensemble_score,
                'label': self._get_sentiment_label(ensemble_score),
                'intensity': self._get_intensity_level(ensemble_score)
            }

            # Emotion detection
            results['emotion'] = self._detect_emotion(text, ensemble_score)

            # Urgency score (for escalation)
            results['urgency'] = self._calculate_urgency(text, ensemble_score)

            logger.debug(f"Sentiment analysis complete: {results['ensemble']['label']}")

            return results

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._empty_result()

    def _calculate_ensemble(self, results: Dict) -> float:
        """Calculate weighted ensemble sentiment score"""
        scores = []
        weights = []

        # Transformer score (weight: 0.4)
        if 'transformer' in results:
            transformer_score = results['transformer']['score']
            if results['transformer']['label'] == 'NEGATIVE':
                transformer_score = -transformer_score
            scores.append(transformer_score)
            weights.append(0.4)

        # VADER compound score (weight: 0.35)
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            weights.append(0.35)

        # TextBlob polarity (weight: 0.25)
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            weights.append(0.25)

        # Calculate weighted average
        if scores:
            ensemble = np.average(scores, weights=weights)
            return float(ensemble)

        return 0.0

    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score"""
        if score <= self.thresholds['very_negative']:
            return 'very_negative'
        elif score <= self.thresholds['negative']:
            return 'negative'
        elif score <= self.thresholds['neutral']:
            return 'neutral'
        elif score <= self.thresholds['positive']:
            return 'positive'
        else:
            return 'very_positive'

    def _get_intensity_level(self, score: float) -> str:
        """Get intensity level"""
        abs_score = abs(score)
        if abs_score < 0.2:
            return 'weak'
        elif abs_score < 0.5:
            return 'moderate'
        elif abs_score < 0.8:
            return 'strong'
        else:
            return 'very_strong'

    def _detect_emotion(self, text: str, sentiment_score: float) -> Dict:
        """
        Detect primary emotion from text

        Returns basic emotion detection based on keywords and sentiment
        """
        text_lower = text.lower()

        # Emotion keyword patterns
        emotions = {
            'angry': ['angry', 'furious', 'mad', 'outraged', 'frustrated', 'annoyed'],
            'sad': ['sad', 'disappointed', 'unhappy', 'depressed', 'upset'],
            'happy': ['happy', 'glad', 'pleased', 'satisfied', 'delighted', 'great'],
            'anxious': ['worried', 'anxious', 'nervous', 'concerned', 'afraid'],
            'confused': ['confused', 'unclear', 'don\'t understand', 'what do you mean']
        }

        detected_emotions = {}
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                detected_emotions[emotion] = score

        # Primary emotion
        if detected_emotions:
            primary_emotion = max(detected_emotions, key=detected_emotions.get)
            confidence = detected_emotions[primary_emotion] / len(text.split())
        else:
            # Fallback based on sentiment
            if sentiment_score < -0.5:
                primary_emotion = 'angry' if sentiment_score < -0.7 else 'sad'
            elif sentiment_score > 0.5:
                primary_emotion = 'happy'
            else:
                primary_emotion = 'neutral'
            confidence = abs(sentiment_score)

        return {
            'primary': primary_emotion,
            'confidence': min(confidence, 1.0),
            'detected': detected_emotions
        }

    def _calculate_urgency(self, text: str, sentiment_score: float) -> Dict:
        """
        Calculate urgency score for escalation

        Returns urgency score from 0-10
        """
        urgency_score = 0.0
        factors = []

        # Strong negative sentiment increases urgency
        if sentiment_score < -0.5:
            urgency_score += 3.0
            factors.append('negative_sentiment')

        # Very strong negative sentiment
        if sentiment_score < -0.7:
            urgency_score += 2.0
            factors.append('very_negative_sentiment')

        # Urgency keywords
        urgency_keywords = {
            'high': ['urgent', 'emergency', 'immediately', 'asap', 'critical', 'now'],
            'medium': ['soon', 'quickly', 'important', 'help', 'please'],
            'complaint': ['complaint', 'unacceptable', 'terrible', 'awful', 'worst']
        }

        text_lower = text.lower()

        for level, keywords in urgency_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                if level == 'high':
                    urgency_score += 3.0 * matches
                    factors.append(f'high_urgency_keywords_{matches}')
                elif level == 'medium':
                    urgency_score += 1.5 * matches
                    factors.append(f'medium_urgency_keywords_{matches}')
                elif level == 'complaint':
                    urgency_score += 2.0 * matches
                    factors.append(f'complaint_keywords_{matches}')

        # Exclamation marks and caps
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            urgency_score += 1.0
            factors.append(f'exclamation_marks_{exclamation_count}')

        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5:
            urgency_score += 2.0
            factors.append(f'excessive_caps')

        # Normalize to 0-10 scale
        urgency_score = min(urgency_score, 10.0)

        return {
            'score': round(urgency_score, 2),
            'level': self._get_urgency_level(urgency_score),
            'factors': factors
        }

    def _get_urgency_level(self, score: float) -> str:
        """Get urgency level from score"""
        if score < 3:
            return 'low'
        elif score < 6:
            return 'medium'
        elif score < 8:
            return 'high'
        else:
            return 'critical'

    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'ensemble': {
                'score': 0.0,
                'label': 'neutral',
                'intensity': 'weak'
            },
            'emotion': {
                'primary': 'neutral',
                'confidence': 0.0,
                'detected': {}
            },
            'urgency': {
                'score': 0.0,
                'level': 'low',
                'factors': []
            }
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in batch"""
        return [self.analyze(text) for text in texts]

    def get_sentiment_trend(self, sentiment_history: List[float]) -> Dict:
        """
        Analyze sentiment trend over conversation

        Args:
            sentiment_history: List of sentiment scores over time

        Returns:
            Trend analysis
        """
        if not sentiment_history or len(sentiment_history) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Calculate trend
        initial = np.mean(sentiment_history[:len(sentiment_history)//2])
        recent = np.mean(sentiment_history[len(sentiment_history)//2:])
        change = recent - initial

        # Determine trend direction
        if abs(change) < 0.2:
            trend = 'stable'
        elif change > 0:
            trend = 'improving'
        else:
            trend = 'declining'

        return {
            'trend': trend,
            'change': round(change, 3),
            'initial_avg': round(initial, 3),
            'recent_avg': round(recent, 3)
        }


# Testing function
def test_sentiment_analyzer():
    """Test sentiment analyzer"""
    analyzer = SentimentAnalyzer()

    test_texts = [
        "I'm very happy with your service! Everything works perfectly.",
        "This is terrible. I'm extremely frustrated and angry!",
        "I need help with my account. Can you assist me?",
        "URGENT! This is an emergency! I need immediate help!!!",
        "I'm not sure what to do. I'm confused and worried.",
    ]

    print("Sentiment Analysis Test")
    print("=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        result = analyzer.analyze(text)

        print(f"Sentiment: {result['ensemble']['label']} "
              f"(score: {result['ensemble']['score']:.3f})")
        print(f"Emotion: {result['emotion']['primary']} "
              f"(confidence: {result['emotion']['confidence']:.3f})")
        print(f"Urgency: {result['urgency']['level']} "
              f"(score: {result['urgency']['score']:.1f})")
        print("-" * 60)


if __name__ == "__main__":
    test_sentiment_analyzer()
