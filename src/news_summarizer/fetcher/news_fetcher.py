"""
News Fetcher Module
Fetches news from RSS feeds, news APIs, and web scraping.
"""

import feedparser
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time


class NewsFetcher:
    """Fetch news from various sources."""

    def __init__(self):
        """Initialize news fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Popular RSS feeds
        self.default_feeds = {
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'cnn': 'http://rss.cnn.com/rss/cnn_topstories.rss',
            'reuters': 'http://feeds.reuters.com/reuters/topNews',
            'techcrunch': 'https://techcrunch.com/feed/',
            'ars_technica': 'http://feeds.arstechnica.com/arstechnica/index',
            'hacker_news': 'https://news.ycombinator.com/rss',
            'nyt': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
            'guardian': 'https://www.theguardian.com/world/rss',
        }

    def fetch_rss(self, feed_url: str, max_entries: int = 10) -> Dict:
        """
        Fetch news from RSS feed.

        Args:
            feed_url: RSS feed URL
            max_entries: Maximum number of entries to fetch

        Returns:
            Dictionary with news articles
        """
        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                return {
                    'source': 'rss',
                    'feed_url': feed_url,
                    'articles': [],
                    'error': f'Feed parsing error: {feed.bozo_exception}',
                    'success': False
                }

            articles = []
            for entry in feed.entries[:max_entries]:
                article = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'description': entry.get('description', entry.get('summary', '')),
                    'published': entry.get('published', entry.get('updated', '')),
                    'source': feed.feed.get('title', 'Unknown'),
                    'author': entry.get('author', ''),
                }

                # Clean HTML from description
                if article['description']:
                    article['description'] = self._clean_html(article['description'])

                articles.append(article)

            return {
                'source': 'rss',
                'feed_url': feed_url,
                'feed_title': feed.feed.get('title', ''),
                'articles': articles,
                'total_fetched': len(articles),
                'success': True
            }

        except Exception as e:
            return {
                'source': 'rss',
                'feed_url': feed_url,
                'articles': [],
                'error': str(e),
                'success': False
            }

    def fetch_multiple_feeds(self, feed_urls: List[str],
                            max_entries_per_feed: int = 10) -> Dict:
        """
        Fetch news from multiple RSS feeds.

        Args:
            feed_urls: List of RSS feed URLs
            max_entries_per_feed: Maximum entries per feed

        Returns:
            Dictionary with combined articles
        """
        all_articles = []
        results = {}
        errors = []

        for feed_url in feed_urls:
            result = self.fetch_rss(feed_url, max_entries_per_feed)

            if result['success']:
                all_articles.extend(result['articles'])
                results[feed_url] = {
                    'feed_title': result.get('feed_title', ''),
                    'count': len(result['articles'])
                }
            else:
                errors.append({
                    'feed_url': feed_url,
                    'error': result.get('error', 'Unknown error')
                })

        # Sort by published date (most recent first)
        all_articles.sort(
            key=lambda x: x.get('published', ''),
            reverse=True
        )

        return {
            'source': 'multiple_rss',
            'articles': all_articles,
            'total_articles': len(all_articles),
            'feeds_processed': len(feed_urls),
            'successful_feeds': len(results),
            'failed_feeds': len(errors),
            'feed_results': results,
            'errors': errors if errors else None
        }

    def fetch_default_feeds(self, categories: Optional[List[str]] = None,
                           max_entries_per_feed: int = 10) -> Dict:
        """
        Fetch news from default RSS feeds.

        Args:
            categories: List of categories to fetch (e.g., ['bbc', 'cnn'])
                       If None, fetches from all default feeds
            max_entries_per_feed: Maximum entries per feed

        Returns:
            Dictionary with articles from default feeds
        """
        if categories is None:
            feed_urls = list(self.default_feeds.values())
        else:
            feed_urls = [self.default_feeds[cat] for cat in categories
                        if cat in self.default_feeds]

        if not feed_urls:
            return {
                'source': 'default_feeds',
                'articles': [],
                'error': 'No valid categories specified',
                'success': False
            }

        result = self.fetch_multiple_feeds(feed_urls, max_entries_per_feed)
        result['source'] = 'default_feeds'
        result['categories'] = categories or list(self.default_feeds.keys())

        return result

    def fetch_newsapi(self, api_key: str, query: Optional[str] = None,
                     category: Optional[str] = None,
                     language: str = 'en',
                     page_size: int = 20) -> Dict:
        """
        Fetch news from NewsAPI (requires API key).

        Args:
            api_key: NewsAPI key (get from https://newsapi.org)
            query: Search query
            category: News category (business, technology, etc.)
            language: Language code (default: 'en')
            page_size: Number of articles to fetch

        Returns:
            Dictionary with news articles
        """
        try:
            base_url = 'https://newsapi.org/v2/'

            if query:
                endpoint = 'everything'
                params = {
                    'q': query,
                    'language': language,
                    'pageSize': page_size,
                    'sortBy': 'publishedAt',
                    'apiKey': api_key
                }
            else:
                endpoint = 'top-headlines'
                params = {
                    'language': language,
                    'pageSize': page_size,
                    'apiKey': api_key
                }
                if category:
                    params['category'] = category

            response = self.session.get(
                f'{base_url}{endpoint}',
                params=params,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            if data['status'] != 'ok':
                return {
                    'source': 'newsapi',
                    'articles': [],
                    'error': data.get('message', 'Unknown error'),
                    'success': False
                }

            articles = []
            for article in data['articles']:
                articles.append({
                    'title': article.get('title', ''),
                    'link': article.get('url', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'image_url': article.get('urlToImage', '')
                })

            return {
                'source': 'newsapi',
                'query': query,
                'category': category,
                'articles': articles,
                'total_results': data.get('totalResults', 0),
                'total_fetched': len(articles),
                'success': True
            }

        except requests.exceptions.RequestException as e:
            return {
                'source': 'newsapi',
                'articles': [],
                'error': f'Request error: {str(e)}',
                'success': False
            }
        except Exception as e:
            return {
                'source': 'newsapi',
                'articles': [],
                'error': str(e),
                'success': False
            }

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def search_news(self, query: str, sources: List[str] = None,
                   max_results: int = 20) -> Dict:
        """
        Search for news articles across sources.

        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum results to return

        Returns:
            Dictionary with search results
        """
        # For now, search RSS feeds
        if sources is None:
            sources = list(self.default_feeds.keys())

        feed_urls = [self.default_feeds[source] for source in sources
                    if source in self.default_feeds]

        result = self.fetch_multiple_feeds(feed_urls, max_results)

        # Filter by query
        if query and result.get('articles'):
            query_lower = query.lower()
            filtered = [
                article for article in result['articles']
                if query_lower in article.get('title', '').lower()
                or query_lower in article.get('description', '').lower()
            ]
            result['articles'] = filtered
            result['total_articles'] = len(filtered)

        result['query'] = query
        return result

    def get_trending_topics(self, articles: List[Dict],
                           top_n: int = 10) -> List[str]:
        """
        Extract trending topics from articles.

        Args:
            articles: List of article dictionaries
            top_n: Number of top topics to return

        Returns:
            List of trending topic keywords
        """
        from collections import Counter
        import re

        # Extract words from titles and descriptions
        words = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            # Extract words (3+ characters)
            words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))

        # Common stop words to exclude
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'with',
            'from', 'this', 'that', 'have', 'has', 'was', 'were',
            'been', 'will', 'can', 'said', 'says', 'more', 'than'
        }

        # Count and filter
        word_counts = Counter(
            word for word in words if word not in stop_words
        )

        return [word for word, count in word_counts.most_common(top_n)]

    def get_available_feeds(self) -> Dict[str, str]:
        """Get dictionary of available default feeds."""
        return self.default_feeds.copy()
