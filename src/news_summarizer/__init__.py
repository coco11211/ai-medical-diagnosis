"""
AI News Summarizer
A comprehensive news summarization tool with extractive/abstractive summarization,
sentiment analysis, and topic clustering capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI News Summarizer Team"

from .summarization.extractive import ExtractiveSummarizer
from .summarization.abstractive import AbstractiveSummarizer
from .sentiment.analyzer import SentimentAnalyzer
from .clustering.topic_clustering import TopicClusterer
from .fetcher.news_fetcher import NewsFetcher

__all__ = [
    'ExtractiveSummarizer',
    'AbstractiveSummarizer',
    'SentimentAnalyzer',
    'TopicClusterer',
    'NewsFetcher'
]
