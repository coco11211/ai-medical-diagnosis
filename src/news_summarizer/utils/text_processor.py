"""
Text Processing Utilities
Common text processing functions for the news summarizer.
"""

import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class TextProcessor:
    """Text processing utilities."""

    def __init__(self):
        """Initialize text processor."""
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded."""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)

        return text.strip()

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            top_n: Number of keywords to extract

        Returns:
            List of keywords
        """
        from collections import Counter
        from nltk.corpus import stopwords

        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()

        # Tokenize and filter
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and len(w) > 3
                and w not in stop_words]

        # Count and return top words
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        try:
            return sent_tokenize(text)
        except:
            # Fallback to simple split
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    def get_reading_time(self, text: str, wpm: int = 200) -> int:
        """
        Estimate reading time in minutes.

        Args:
            text: Input text
            wpm: Words per minute reading speed

        Returns:
            Reading time in minutes
        """
        word_count = len(text.split())
        return max(1, round(word_count / wpm))

    def get_text_stats(self, text: str) -> Dict:
        """
        Get text statistics.

        Args:
            text: Input text

        Returns:
            Dictionary with text statistics
        """
        sentences = self.split_into_sentences(text)
        words = text.split()

        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'reading_time_minutes': self.get_reading_time(text)
        }

    def truncate_text(self, text: str, max_length: int = 500,
                     suffix: str = '...') -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add when truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix

    def highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """
        Highlight keywords in text (for console output).

        Args:
            text: Input text
            keywords: List of keywords to highlight

        Returns:
            Text with highlighted keywords
        """
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(f'**{keyword.upper()}**', text)

        return text
