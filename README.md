# AI News Summarizer

A comprehensive AI-powered news summarization tool featuring extractive/abstractive summarization, sentiment analysis, and topic clustering capabilities. Optimized for Windows 11 with both GUI and CLI interfaces.

## Features

### 1. **Dual Summarization Methods**

#### Extractive Summarization
- **TF-IDF**: Statistical approach using term frequency-inverse document frequency
- **TextRank**: Graph-based algorithm inspired by PageRank
- Fast and efficient, works offline
- Extracts the most important sentences from the original text

#### Abstractive Summarization
- **BART** (facebook/bart-large-cnn): Powerful transformer model for news
- **Pegasus**: Specialized for abstractive summarization
- **T5**: Versatile text-to-text transformer
- Generates human-like summaries with paraphrasing
- Supports long document summarization via chunking

### 2. **Sentiment Analysis**

- **VADER**: Rule-based sentiment analysis optimized for social media and news
- **Transformer Models**: Deep learning-based sentiment classification
- Compound sentiment scores with confidence levels
- Aspect-based sentiment analysis
- Emotion detection (joy, sadness, anger, fear, surprise)
- Sentiment trend analysis across multiple articles

### 3. **Topic Clustering**

- **K-Means**: Distance-based clustering with TF-IDF vectors
- **LDA** (Latent Dirichlet Allocation): Probabilistic topic modeling
- **NMF** (Non-negative Matrix Factorization): Linear algebra-based topics
- Automatic optimal cluster detection
- Topic keyword extraction
- Article grouping by topic

### 4. **News Fetching**

- **RSS Feed Support**: Built-in feeds from BBC, CNN, Reuters, TechCrunch, NYT, Guardian, etc.
- **NewsAPI Integration**: Fetch from 70,000+ sources (requires API key)
- **Search Functionality**: Search across multiple sources
- **Trending Topics**: Automatic keyword extraction
- **Multi-source Aggregation**: Combine news from multiple feeds

### 5. **Export Capabilities**

- **JSON**: Structured data export
- **HTML**: Beautiful formatted reports with styling
- **Markdown**: GitHub-compatible documentation
- **Plain Text**: Simple text-based reports
- Batch export of analysis results

### 6. **User Interfaces**

#### GUI (Windows 11 Optimized)
- Modern, clean interface using tkinter
- Windows 11 design language
- Three-tab layout: Fetch, Analyze, Results
- Real-time progress tracking
- Multi-threaded operations for smooth UX
- One-click export functionality

#### CLI
- Comprehensive command-line interface
- Pipeline mode for automated workflows
- Scriptable and automation-friendly
- Detailed progress output

## System Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB+ recommended for transformers)
- **Storage**: 2GB free space (for models)
- **Internet**: Required for fetching news and downloading models

## Installation

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd ai-medical-diagnosis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download NLTK data and transformer models automatically.

### Step 3: (Optional) Get NewsAPI Key

For extended news sources:
1. Visit [https://newsapi.org](https://newsapi.org)
2. Sign up for free API key
3. Use with `--api-key` parameter

## Quick Start

### GUI Application

Launch the graphical interface:

```bash
python news_summarizer.py gui
```

**GUI Workflow:**
1. **Fetch Tab**: Select news sources and fetch articles
2. **Analyze Tab**: Choose analysis options (summarization, sentiment, clustering)
3. **Results Tab**: View results and export to various formats

### CLI Commands

#### 1. Fetch News

Fetch from default sources:
```bash
python news_summarizer.py fetch --max-articles 20
```

Fetch from specific sources:
```bash
python news_summarizer.py fetch --sources bbc cnn techcrunch --max-articles 10
```

Export to file:
```bash
python news_summarizer.py fetch --output news.json
```

#### 2. Summarize Text

Extractive summarization:
```bash
python news_summarizer.py summarize --text "Your long text here..." --method extractive --sentences 3
```

Abstractive summarization:
```bash
python news_summarizer.py summarize --file article.txt --method abstractive --model facebook/bart-large-cnn
```

Both methods:
```bash
python news_summarizer.py summarize --file article.txt --method both
```

#### 3. Sentiment Analysis

Analyze sentiment with VADER:
```bash
python news_summarizer.py sentiment --text "This is amazing news!" --method vader
```

Use transformer model:
```bash
python news_summarizer.py sentiment --file article.txt --method transformer
```

#### 4. Topic Clustering

Cluster articles from JSON file:
```bash
python news_summarizer.py cluster --file news.json --topics 5 --method lda
```

Different clustering methods:
```bash
python news_summarizer.py cluster --file news.json --method kmeans --topics 3
```

#### 5. Full Pipeline

Run complete analysis (fetch + summarize + sentiment + cluster + export):

```bash
python news_summarizer.py pipeline --max-articles 30 --output analysis.html
```

Specific sources:
```bash
python news_summarizer.py pipeline --sources bbc reuters techcrunch --output tech_news.html
```

## Available News Sources

Default RSS feeds included:
- **bbc**: BBC News
- **cnn**: CNN Top Stories
- **reuters**: Reuters Top News
- **techcrunch**: TechCrunch
- **ars_technica**: Ars Technica
- **hacker_news**: Hacker News
- **nyt**: New York Times
- **guardian**: The Guardian

## Configuration Options

### Summarization Parameters

**Extractive:**
- `--sentences`: Number of sentences (default: 3)
- `--algorithm`: tfidf or textrank (default: textrank)

**Abstractive:**
- `--model`: Transformer model (default: facebook/bart-large-cnn)
- `--max-length`: Maximum summary length (default: 130)
- `--min-length`: Minimum summary length (default: 30)

**Available Models:**
- `facebook/bart-large-cnn` (Best for news, balanced)
- `google/pegasus-xsum` (Shorter summaries)
- `google/pegasus-cnn_dailymail` (News-focused)
- `t5-base` (Versatile, lighter)

### Sentiment Analysis

**Methods:**
- `vader`: Fast, rule-based (offline)
- `transformer`: Deep learning-based (requires model download)
- `both`: Run both methods

### Topic Clustering

**Methods:**
- `lda`: Latent Dirichlet Allocation (best for topics)
- `kmeans`: K-Means clustering (fast)
- `nmf`: Non-negative Matrix Factorization (interpretable)

**Parameters:**
- `--topics`: Number of topics/clusters (default: 5)

## Project Structure

```
ai-news-summarizer/
├── src/
│   └── news_summarizer/
│       ├── __init__.py
│       ├── summarization/
│       │   ├── extractive.py      # TF-IDF & TextRank
│       │   └── abstractive.py     # Transformer models
│       ├── sentiment/
│       │   └── analyzer.py        # VADER & transformers
│       ├── clustering/
│       │   └── topic_clustering.py # K-Means, LDA, NMF
│       ├── fetcher/
│       │   └── news_fetcher.py    # RSS & NewsAPI
│       ├── gui/
│       │   └── main_window.py     # Windows 11 GUI
│       └── utils/
│           ├── text_processor.py  # Text utilities
│           └── export.py          # Export manager
├── news_summarizer.py             # Main CLI interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Usage Examples

### Example 1: Quick News Summary

```bash
# Fetch and summarize tech news
python news_summarizer.py pipeline --sources techcrunch ars_technica --max-articles 10 --output tech_summary.html

# Open tech_summary.html in browser to view beautiful formatted report
```

### Example 2: Sentiment Analysis of Articles

```bash
# Fetch news
python news_summarizer.py fetch --sources cnn bbc --output news.json

# Analyze sentiment
python news_summarizer.py sentiment --file news.json --method both --output sentiment.json
```

### Example 3: Topic Discovery

```bash
# Fetch diverse news
python news_summarizer.py fetch --sources bbc cnn reuters guardian nyt --max-articles 50 --output diverse_news.json

# Discover topics
python news_summarizer.py cluster --file diverse_news.json --topics 10 --method lda --output topics.json
```

### Example 4: Custom Text Summarization

```bash
# Create a text file with your content
echo "Your long article text here..." > article.txt

# Get both extractive and abstractive summaries
python news_summarizer.py summarize --file article.txt --method both --sentences 5
```

## Performance Tips

### For Faster Processing:
- Use **extractive** summarization instead of abstractive
- Use **VADER** sentiment analysis instead of transformers
- Limit `--max-articles` to reasonable numbers (20-50)
- Use lighter models like `t5-small` for abstractive summarization

### For Better Quality:
- Use **abstractive** summarization with `facebook/bart-large-cnn`
- Use **both** sentiment methods for more accurate analysis
- Increase `--topics` for finer-grained clustering
- Use **LDA** for topic modeling (better topics than K-Means)

### For Low Memory Systems:
- Avoid loading transformer models
- Stick with extractive + VADER + K-Means
- Process in smaller batches
- Use `t5-small` if you need abstractive summarization

## Output Formats

### JSON Export
Structured data with all analysis results, perfect for further processing.

### HTML Export
Beautiful, styled report with:
- Color-coded sentiment indicators
- Organized article summaries
- Topic tags
- Responsive design
- Direct links to original articles

### Markdown Export
GitHub-compatible markdown with:
- Hierarchical structure
- Formatted tables
- Easy to read and edit
- Version control friendly

### Plain Text Export
Simple, clean text output for:
- Email reports
- Terminal viewing
- Legacy systems

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Transformer model download fails
**Solution**: Check internet connection or use extractive mode
```bash
python news_summarizer.py summarize --file text.txt --method extractive
```

### Issue: Out of memory when using transformers
**Solution**: Use lighter models or extractive summarization
```bash
# Use lighter model
python news_summarizer.py summarize --file text.txt --method abstractive --model t5-small

# Or use extractive
python news_summarizer.py summarize --file text.txt --method extractive
```

### Issue: RSS feed timeout
**Solution**:
- Check internet connection
- Try fewer sources
- Increase timeout in code if needed

### Issue: GUI doesn't launch
**Solution**: Ensure tkinter is installed (comes with Python on Windows)
```bash
python -m tkinter  # Test tkinter installation
```

## Advanced Usage

### Using NewsAPI

```python
from news_summarizer import NewsFetcher

fetcher = NewsFetcher()
result = fetcher.fetch_newsapi(
    api_key='YOUR_API_KEY',
    query='artificial intelligence',
    language='en',
    page_size=50
)
```

### Custom Pipelines

```python
from news_summarizer import (
    ExtractiveSummarizer,
    SentimentAnalyzer,
    NewsFetcher
)

# Fetch news
fetcher = NewsFetcher()
articles = fetcher.fetch_default_feeds()

# Analyze each article
summarizer = ExtractiveSummarizer()
analyzer = SentimentAnalyzer()

for article in articles['articles']:
    text = f"{article['title']} {article['description']}"

    # Summarize
    summary = summarizer.summarize(text)

    # Sentiment
    sentiment = analyzer.analyze(text)

    print(f"Title: {article['title']}")
    print(f"Summary: {summary['summary']}")
    print(f"Sentiment: {sentiment['sentiment']}")
    print("-" * 80)
```

### Batch Processing

```python
from news_summarizer import AbstractiveSummarizer

summarizer = AbstractiveSummarizer()

texts = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]
summaries = summarizer.batch_summarize(texts)

for i, summary in enumerate(summaries):
    print(f"Summary {i+1}: {summary['summary']}")
```

## API Reference

### ExtractiveSummarizer
- `summarize(text, method='textrank', num_sentences=3)`: Summarize text
- `tfidf_summarize(text, num_sentences=3)`: TF-IDF based summary
- `textrank_summarize(text, num_sentences=3)`: TextRank summary
- `get_key_sentences(text, top_k=5)`: Extract key sentences with scores

### AbstractiveSummarizer
- `summarize(text, max_length=130, min_length=30)`: Generate summary
- `summarize_long_text(text, chunk_size=1024)`: Handle long documents
- `batch_summarize(texts)`: Summarize multiple texts
- `unload_model()`: Free memory

### SentimentAnalyzer
- `analyze(text, method='vader')`: Analyze sentiment
- `analyze_vader(text)`: VADER analysis
- `analyze_transformer(text)`: Transformer analysis
- `get_aspect_sentiment(text, aspects)`: Aspect-based analysis
- `analyze_sentiment_trend(texts)`: Trend analysis

### TopicClusterer
- `kmeans_cluster(texts, n_clusters=5)`: K-Means clustering
- `lda_topics(texts, n_topics=5)`: LDA topic modeling
- `nmf_topics(texts, n_topics=5)`: NMF topic modeling
- `find_optimal_clusters(texts)`: Find optimal cluster count

### NewsFetcher
- `fetch_rss(feed_url)`: Fetch from RSS feed
- `fetch_default_feeds()`: Fetch from all default feeds
- `fetch_newsapi(api_key, query)`: Fetch from NewsAPI
- `search_news(query, sources)`: Search across sources
- `get_trending_topics(articles)`: Extract trending topics

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

Built with:
- **Python** - Core language
- **NLTK** - Natural language processing
- **scikit-learn** - Machine learning algorithms
- **Transformers** (Hugging Face) - State-of-the-art NLP models
- **NetworkX** - Graph algorithms for TextRank
- **feedparser** - RSS feed parsing
- **tkinter** - GUI framework

Special thanks to:
- Hugging Face for pre-trained models
- NLTK project for language resources
- News outlets for RSS feeds

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the API reference

## Citation

If you use this tool in research, please cite:
```
AI News Summarizer - A comprehensive news analysis tool with extractive/abstractive
summarization, sentiment analysis, and topic clustering.
```

---

**Happy News Summarizing!**

*Stay informed with AI-powered insights.*
