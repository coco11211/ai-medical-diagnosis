#!/usr/bin/env python3
"""
AI News Summarizer - Main CLI Interface
Comprehensive news summarization with extractive/abstractive summarization,
sentiment analysis, and topic clustering.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from news_summarizer import (
    ExtractiveSummarizer,
    AbstractiveSummarizer,
    SentimentAnalyzer,
    TopicClusterer,
    NewsFetcher
)
from news_summarizer.utils.text_processor import TextProcessor
from news_summarizer.utils.export import ExportManager


def fetch_command(args):
    """Fetch news from sources."""
    fetcher = NewsFetcher()

    print(f"\nFetching news from sources...")

    if args.sources:
        # Fetch from specified sources
        available_feeds = fetcher.get_available_feeds()
        feed_urls = [available_feeds[source] for source in args.sources
                    if source in available_feeds]

        if not feed_urls:
            print("Error: No valid sources specified")
            print(f"Available sources: {', '.join(available_feeds.keys())}")
            return

        result = fetcher.fetch_multiple_feeds(feed_urls, args.max_articles)
    else:
        # Fetch from all default sources
        result = fetcher.fetch_default_feeds(max_entries_per_feed=args.max_articles)

    if not result.get('success', False):
        print(f"Error fetching news: {result.get('error', 'Unknown error')}")
        return

    articles = result.get('articles', [])
    print(f"\nFetched {len(articles)} articles from {result.get('successful_feeds', 0)} sources\n")

    # Display articles
    for i, article in enumerate(articles[:args.max_articles], 1):
        print(f"{i}. {article.get('title', 'Untitled')}")
        print(f"   Source: {article.get('source', 'Unknown')} | Published: {article.get('published', 'N/A')}")
        if args.show_description:
            print(f"   {article.get('description', '')[:200]}...")
        print()

    # Save to file if requested
    if args.output:
        export_manager = ExportManager()
        data = {'articles': articles}
        if export_manager.export(data, args.output):
            print(f"\nArticles saved to {args.output}")


def summarize_command(args):
    """Summarize text or articles."""
    # Read input
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Please provide text with --text or --file")
        return

    print(f"\nSummarizing text using {args.method} method...")

    # Perform summarization
    if args.method in ['extractive', 'both']:
        print("\nExtractive Summary:")
        print("-" * 80)
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(
            text,
            method='textrank' if args.algorithm == 'textrank' else 'tfidf',
            num_sentences=args.sentences
        )
        print(result['summary'])
        print(f"\n(Compressed from {result['original_length']} to "
              f"{result['summary_length']} sentences)")

    if args.method in ['abstractive', 'both']:
        print("\nAbstractive Summary:")
        print("-" * 80)
        print("Loading transformer model...")
        summarizer = AbstractiveSummarizer(model_name=args.model)
        result = summarizer.summarize(
            text,
            max_length=args.max_length,
            min_length=args.min_length
        )
        print(result['summary'])
        if 'compression_ratio' in result:
            print(f"\n(Compression ratio: {result['compression_ratio']:.2%})")

    # Save if requested
    if args.output:
        export_manager = ExportManager()
        data = {
            'original_text': text,
            'summaries': result,
            'method': args.method
        }
        if export_manager.export(data, args.output):
            print(f"\nSummary saved to {args.output}")


def sentiment_command(args):
    """Analyze sentiment of text."""
    # Read input
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Please provide text with --text or --file")
        return

    print("\nAnalyzing sentiment...")

    analyzer = SentimentAnalyzer(
        use_transformer=args.method == 'transformer',
        transformer_model=args.model
    )

    result = analyzer.analyze(text, method=args.method)

    print("\nSentiment Analysis Results:")
    print("-" * 80)
    print(f"Sentiment: {result.get('sentiment', 'N/A').upper()}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")

    if args.method == 'vader' and 'scores' in result:
        scores = result['scores']
        print(f"\nDetailed Scores:")
        print(f"  Positive: {scores.get('pos', 0):.2%}")
        print(f"  Negative: {scores.get('neg', 0):.2%}")
        print(f"  Neutral: {scores.get('neu', 0):.2%}")
        print(f"  Compound: {scores.get('compound', 0):.3f}")

    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cluster_command(args):
    """Cluster articles by topic."""
    # Read articles from file
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'articles' in data:
                articles = data['articles']
                texts = [f"{a.get('title', '')} {a.get('description', '')}"
                        for a in articles]
            elif isinstance(data, list):
                texts = data
            else:
                print("Error: Invalid file format")
                return
    else:
        print("Error: Please provide input file with --file")
        return

    if len(texts) < args.topics:
        print(f"Warning: Only {len(texts)} texts available, reducing topics to {len(texts)}")
        args.topics = max(2, len(texts) - 1)

    print(f"\nClustering {len(texts)} texts into {args.topics} topics using {args.method}...")

    clusterer = TopicClusterer(n_topics=args.topics)
    result = clusterer.cluster_and_analyze(texts, method=args.method)

    if not result.get('success', False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print("\nTopic Clustering Results:")
    print("=" * 80)

    if args.method == 'kmeans':
        for cluster_id, cluster_info in result['cluster_terms'].items():
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Top terms: {', '.join(cluster_info['terms'][:10])}")
            print(f"  Articles: {len(result['clustered_texts'].get(cluster_id, []))}")
    else:
        for topic_id, topic_info in result['topics'].items():
            print(f"\nTopic {topic_id + 1}:")
            print(f"  Keywords: {', '.join(topic_info['terms'][:10])}")
            if 'topic_texts' in result:
                print(f"  Articles: {len(result['topic_texts'].get(topic_id, []))}")

    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


def gui_command(args):
    """Launch GUI application."""
    print("Launching News Summarizer GUI...")
    from news_summarizer.gui.main_window import main
    main()


def pipeline_command(args):
    """Run full analysis pipeline."""
    print("\nRunning full news analysis pipeline...")
    print("=" * 80)

    # Step 1: Fetch news
    print("\n1. Fetching news articles...")
    fetcher = NewsFetcher()

    if args.sources:
        available_feeds = fetcher.get_available_feeds()
        feed_urls = [available_feeds[source] for source in args.sources
                    if source in available_feeds]
        result = fetcher.fetch_multiple_feeds(feed_urls, args.max_articles)
    else:
        result = fetcher.fetch_default_feeds(max_entries_per_feed=args.max_articles)

    if not result.get('success', False):
        print(f"Error fetching news: {result.get('error', 'Unknown error')}")
        return

    articles = result.get('articles', [])
    print(f"   Fetched {len(articles)} articles")

    # Step 2: Summarize
    print("\n2. Summarizing articles...")
    extractive_summarizer = ExtractiveSummarizer()
    sentiment_analyzer = SentimentAnalyzer()

    processed_articles = []

    for i, article in enumerate(articles, 1):
        print(f"   Processing article {i}/{len(articles)}...", end='\r')

        text = f"{article.get('title', '')} {article.get('description', '')}"

        # Summarize
        summary = extractive_summarizer.summarize(text, num_sentences=3)

        # Sentiment
        sentiment = sentiment_analyzer.analyze(text, method='vader')

        processed_articles.append({
            'title': article.get('title', ''),
            'source': article.get('source', ''),
            'published': article.get('published', ''),
            'link': article.get('link', ''),
            'summary': summary,
            'sentiment': sentiment
        })

    print(f"   Processed {len(processed_articles)} articles")

    # Step 3: Topic clustering
    if len(articles) >= 2:
        print("\n3. Clustering topics...")
        texts = [f"{a.get('title', '')} {a.get('description', '')}"
                for a in articles]

        clusterer = TopicClusterer(n_topics=min(5, len(texts) - 1))
        clustering = clusterer.lda_topics(texts)
        print(f"   Identified {len(clustering.get('topics', {}))} topics")
    else:
        clustering = None

    # Step 4: Export results
    print("\n4. Exporting results...")
    export_manager = ExportManager()

    output_data = {
        'articles': processed_articles,
        'clustering': clustering
    }

    if export_manager.export(output_data, args.output):
        print(f"   Results exported to {args.output}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print(f"\nSummary:")
    print(f"  - {len(articles)} articles processed")
    print(f"  - Sentiment distribution:")

    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    for article in processed_articles:
        sent = article['sentiment'].get('sentiment', 'neutral')
        sentiments[sent] = sentiments.get(sent, 0) + 1

    for sent, count in sentiments.items():
        print(f"    {sent.capitalize()}: {count} ({count/len(articles)*100:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI News Summarizer - Extractive/Abstractive summarization, '
                   'sentiment analysis, and topic clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch news articles')
    fetch_parser.add_argument('--sources', nargs='+', help='News sources to fetch from')
    fetch_parser.add_argument('--max-articles', type=int, default=10,
                            help='Maximum articles per source')
    fetch_parser.add_argument('--show-description', action='store_true',
                            help='Show article descriptions')
    fetch_parser.add_argument('--output', '-o', help='Output file (JSON/HTML/TXT)')

    # Summarize command
    sum_parser = subparsers.add_parser('summarize', help='Summarize text')
    sum_parser.add_argument('--text', help='Text to summarize')
    sum_parser.add_argument('--file', help='File containing text to summarize')
    sum_parser.add_argument('--method', choices=['extractive', 'abstractive', 'both'],
                          default='extractive', help='Summarization method')
    sum_parser.add_argument('--algorithm', choices=['tfidf', 'textrank'],
                          default='textrank', help='Extractive algorithm')
    sum_parser.add_argument('--sentences', type=int, default=3,
                          help='Number of sentences in extractive summary')
    sum_parser.add_argument('--model', default='facebook/bart-large-cnn',
                          help='Transformer model for abstractive summarization')
    sum_parser.add_argument('--max-length', type=int, default=130,
                          help='Max length for abstractive summary')
    sum_parser.add_argument('--min-length', type=int, default=30,
                          help='Min length for abstractive summary')
    sum_parser.add_argument('--output', '-o', help='Output file')

    # Sentiment command
    sent_parser = subparsers.add_parser('sentiment', help='Analyze sentiment')
    sent_parser.add_argument('--text', help='Text to analyze')
    sent_parser.add_argument('--file', help='File containing text to analyze')
    sent_parser.add_argument('--method', choices=['vader', 'transformer', 'both'],
                           default='vader', help='Sentiment analysis method')
    sent_parser.add_argument('--model', default='distilbert-base-uncased-finetuned-sst-2-english',
                           help='Transformer model for sentiment analysis')
    sent_parser.add_argument('--output', '-o', help='Output file')

    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Cluster articles by topic')
    cluster_parser.add_argument('--file', required=True, help='JSON file with articles')
    cluster_parser.add_argument('--topics', type=int, default=5, help='Number of topics')
    cluster_parser.add_argument('--method', choices=['kmeans', 'lda', 'nmf'],
                              default='lda', help='Clustering method')
    cluster_parser.add_argument('--output', '-o', help='Output file')

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline',
                                           help='Run full analysis pipeline')
    pipeline_parser.add_argument('--sources', nargs='+', help='News sources')
    pipeline_parser.add_argument('--max-articles', type=int, default=20,
                               help='Maximum articles to fetch')
    pipeline_parser.add_argument('--output', '-o', default='news_analysis.html',
                               help='Output file (HTML/JSON/TXT/MD)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate command
    if args.command == 'fetch':
        fetch_command(args)
    elif args.command == 'summarize':
        summarize_command(args)
    elif args.command == 'sentiment':
        sentiment_command(args)
    elif args.command == 'cluster':
        cluster_command(args)
    elif args.command == 'gui':
        gui_command(args)
    elif args.command == 'pipeline':
        pipeline_command(args)


if __name__ == '__main__':
    main()
