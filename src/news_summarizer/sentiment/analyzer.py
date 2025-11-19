"""
Sentiment Analysis Module
Implements sentiment analysis using VADER and transformer-based models.
"""

from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Sentiment analysis using VADER and transformer models."""

    def __init__(self, use_transformer: bool = False,
                 transformer_model: str = 'distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize sentiment analyzer.

        Args:
            use_transformer: Whether to use transformer model
            transformer_model: Hugging Face model for sentiment analysis
        """
        self.use_transformer = use_transformer
        self.transformer_model = transformer_model
        self.vader_analyzer = None
        self.transformer_pipeline = None
        self._vader_loaded = False
        self._transformer_loaded = False

    def _load_vader(self):
        """Load VADER sentiment analyzer."""
        if self._vader_loaded:
            return

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk

            # Download VADER lexicon
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)

            self.vader_analyzer = SentimentIntensityAnalyzer()
            self._vader_loaded = True

        except ImportError:
            raise ImportError(
                "NLTK not installed. Install with: pip install nltk"
            )

    def _load_transformer(self):
        """Load transformer-based sentiment model."""
        if self._transformer_loaded:
            return

        try:
            from transformers import pipeline

            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=self.transformer_model,
                device=-1  # Use CPU by default
            )
            self._transformer_loaded = True

        except ImportError:
            raise ImportError(
                "Transformers library not installed. "
                "Install with: pip install transformers torch"
            )

    def analyze_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER.

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores and classification
        """
        self._load_vader()

        scores = self.vader_analyzer.polarity_scores(text)

        # Classify sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'method': 'vader',
            'sentiment': sentiment,
            'scores': scores,
            'compound': compound,
            'confidence': abs(compound)
        }

    def analyze_transformer(self, text: str, truncate: bool = True) -> Dict:
        """
        Analyze sentiment using transformer model.

        Args:
            text: Input text
            truncate: Whether to truncate long texts

        Returns:
            Dictionary with sentiment and confidence
        """
        self._load_transformer()

        try:
            result = self.transformer_pipeline(text, truncation=truncate)[0]

            return {
                'method': 'transformer',
                'model': self.transformer_model,
                'sentiment': result['label'].lower(),
                'confidence': result['score'],
                'raw_result': result
            }

        except Exception as e:
            return {
                'method': 'transformer',
                'model': self.transformer_model,
                'sentiment': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }

    def analyze(self, text: str, method: str = 'vader') -> Dict:
        """
        Analyze sentiment using specified method.

        Args:
            text: Input text
            method: 'vader', 'transformer', or 'both'

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'method': method,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'Empty text'
            }

        if method == 'vader':
            return self.analyze_vader(text)
        elif method == 'transformer':
            return self.analyze_transformer(text)
        elif method == 'both':
            vader_result = self.analyze_vader(text)
            transformer_result = self.analyze_transformer(text)

            return {
                'method': 'combined',
                'vader': vader_result,
                'transformer': transformer_result,
                'sentiment': vader_result['sentiment'],  # Default to VADER
                'confidence': (vader_result['confidence'] +
                             transformer_result['confidence']) / 2
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_batch(self, texts: List[str], method: str = 'vader') -> List[Dict]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            method: Analysis method to use

        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text, method) for text in texts]

    def get_aspect_sentiment(self, text: str, aspects: List[str]) -> Dict:
        """
        Analyze sentiment for specific aspects in text.

        Args:
            text: Input text
            aspects: List of aspects to analyze

        Returns:
            Dictionary with aspect-specific sentiments
        """
        self._load_vader()

        results = {}
        sentences = text.split('.')

        for aspect in aspects:
            aspect_sentences = [s for s in sentences
                              if aspect.lower() in s.lower()]

            if aspect_sentences:
                combined_text = ' '.join(aspect_sentences)
                sentiment = self.analyze_vader(combined_text)
                results[aspect] = sentiment
            else:
                results[aspect] = {
                    'sentiment': 'not_found',
                    'confidence': 0.0
                }

        return {
            'method': 'aspect_based',
            'aspects': results,
            'overall': self.analyze_vader(text)
        }

    def get_emotion_scores(self, text: str) -> Dict:
        """
        Get detailed emotion scores from text.

        Args:
            text: Input text

        Returns:
            Dictionary with emotion scores
        """
        # Use VADER as base
        vader_result = self.analyze_vader(text)

        # Simple emotion mapping based on VADER scores
        pos = vader_result['scores']['pos']
        neg = vader_result['scores']['neg']
        neu = vader_result['scores']['neu']

        emotions = {
            'joy': pos * 0.7 if pos > 0.3 else 0,
            'sadness': neg * 0.6 if neg > 0.3 else 0,
            'anger': neg * 0.4 if neg > 0.5 else 0,
            'fear': neg * 0.3 if neg > 0.4 else 0,
            'surprise': abs(vader_result['compound']) * 0.5
                       if abs(vader_result['compound']) > 0.5 else 0,
            'neutral': neu
        }

        return {
            'method': 'emotion_detection',
            'emotions': emotions,
            'dominant_emotion': max(emotions, key=emotions.get),
            'base_sentiment': vader_result
        }

    def analyze_sentiment_trend(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment trend across multiple texts.

        Args:
            texts: List of texts in chronological order

        Returns:
            Dictionary with trend analysis
        """
        sentiments = []
        scores = []

        for text in texts:
            result = self.analyze_vader(text)
            sentiments.append(result['sentiment'])
            scores.append(result['compound'])

        # Calculate statistics
        avg_score = sum(scores) / len(scores) if scores else 0
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }

        # Determine trend
        if len(scores) >= 2:
            early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
            late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            trend = 'improving' if late_avg > early_avg else 'declining' if late_avg < early_avg else 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'method': 'trend_analysis',
            'num_texts': len(texts),
            'average_sentiment_score': avg_score,
            'sentiment_distribution': sentiment_counts,
            'trend': trend,
            'scores': scores,
            'sentiments': sentiments
        }
