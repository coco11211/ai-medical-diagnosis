"""
Extractive Summarization Module
Implements TF-IDF, TextRank, and other extractive summarization techniques.
"""

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Dict, Tuple


class ExtractiveSummarizer:
    """Extractive text summarization using multiple algorithms."""

    def __init__(self):
        """Initialize the extractive summarizer."""
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass  # Handle offline mode

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()

    def _sentence_similarity(self, sent1: str, sent2: str,
                            vectorizer: TfidfVectorizer) -> float:
        """Calculate cosine similarity between two sentences."""
        try:
            vectors = vectorizer.fit_transform([sent1, sent2])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            return 0.0

    def tfidf_summarize(self, text: str, num_sentences: int = 3,
                       min_sentence_length: int = 10) -> Dict:
        """
        Summarize text using TF-IDF scoring.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            min_sentence_length: Minimum sentence length to consider

        Returns:
            Dictionary with summary and metadata
        """
        text = self._preprocess_text(text)
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'sentences': sentences,
                'method': 'tfidf',
                'original_length': len(sentences),
                'summary_length': len(sentences)
            }

        # Filter short sentences
        valid_sentences = [(i, s) for i, s in enumerate(sentences)
                          if len(s.split()) >= min_sentence_length]

        if not valid_sentences:
            valid_sentences = list(enumerate(sentences))

        indices, filtered_sentences = zip(*valid_sentences)

        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(filtered_sentences)

            # Calculate sentence scores (sum of TF-IDF values)
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

            # Get top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

            # Sort by original order to maintain coherence
            selected_indices = sorted([indices[i] for i in top_indices])
            summary_sentences = [sentences[i] for i in selected_indices]

            return {
                'summary': ' '.join(summary_sentences),
                'sentences': summary_sentences,
                'method': 'tfidf',
                'original_length': len(sentences),
                'summary_length': len(summary_sentences),
                'scores': {i: float(sentence_scores[list(indices).index(i)])
                          for i in selected_indices if i in indices}
            }
        except Exception as e:
            # Fallback to first sentences
            return {
                'summary': ' '.join(sentences[:num_sentences]),
                'sentences': sentences[:num_sentences],
                'method': 'tfidf_fallback',
                'original_length': len(sentences),
                'summary_length': num_sentences,
                'error': str(e)
            }

    def textrank_summarize(self, text: str, num_sentences: int = 3,
                          damping: float = 0.85, min_sentence_length: int = 10) -> Dict:
        """
        Summarize text using TextRank algorithm.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            damping: PageRank damping factor
            min_sentence_length: Minimum sentence length to consider

        Returns:
            Dictionary with summary and metadata
        """
        text = self._preprocess_text(text)
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'sentences': sentences,
                'method': 'textrank',
                'original_length': len(sentences),
                'summary_length': len(sentences)
            }

        # Filter short sentences
        valid_sentences = [(i, s) for i, s in enumerate(sentences)
                          if len(s.split()) >= min_sentence_length]

        if not valid_sentences:
            valid_sentences = list(enumerate(sentences))

        indices, filtered_sentences = zip(*valid_sentences)

        try:
            # Build similarity matrix
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Build graph
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, alpha=damping)

            # Rank sentences
            ranked_sentences = sorted(((scores[i], i, s)
                                      for i, s in enumerate(filtered_sentences)),
                                     reverse=True)

            # Get top sentences and sort by original order
            top_sentence_indices = sorted([indices[ranked_sentences[i][1]]
                                          for i in range(min(num_sentences, len(ranked_sentences)))])

            summary_sentences = [sentences[i] for i in top_sentence_indices]

            return {
                'summary': ' '.join(summary_sentences),
                'sentences': summary_sentences,
                'method': 'textrank',
                'original_length': len(sentences),
                'summary_length': len(summary_sentences),
                'scores': {indices[i]: float(score)
                          for score, i, _ in ranked_sentences[:num_sentences]}
            }
        except Exception as e:
            # Fallback to first sentences
            return {
                'summary': ' '.join(sentences[:num_sentences]),
                'sentences': sentences[:num_sentences],
                'method': 'textrank_fallback',
                'original_length': len(sentences),
                'summary_length': num_sentences,
                'error': str(e)
            }

    def summarize(self, text: str, method: str = 'textrank',
                 num_sentences: int = 3, **kwargs) -> Dict:
        """
        Summarize text using specified method.

        Args:
            text: Input text to summarize
            method: 'tfidf' or 'textrank'
            num_sentences: Number of sentences in summary
            **kwargs: Additional arguments for specific methods

        Returns:
            Dictionary with summary and metadata
        """
        if method.lower() == 'tfidf':
            return self.tfidf_summarize(text, num_sentences, **kwargs)
        elif method.lower() == 'textrank':
            return self.textrank_summarize(text, num_sentences, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'textrank'")

    def get_key_sentences(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extract key sentences with their importance scores.

        Args:
            text: Input text
            top_k: Number of key sentences to return

        Returns:
            List of (sentence, score) tuples
        """
        result = self.textrank_summarize(text, num_sentences=top_k)

        if 'scores' in result:
            sentences = sent_tokenize(text)
            scored_sentences = [(sentences[idx], score)
                              for idx, score in result['scores'].items()]
            return sorted(scored_sentences, key=lambda x: x[1], reverse=True)

        return [(s, 1.0) for s in result['sentences']]
