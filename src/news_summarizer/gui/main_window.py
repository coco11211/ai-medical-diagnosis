"""
GUI Main Window for Windows 11
Modern news summarizer interface using tkinter.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from news_summarizer import (
    ExtractiveSummarizer,
    AbstractiveSummarizer,
    SentimentAnalyzer,
    TopicClusterer,
    NewsFetcher
)
from news_summarizer.utils.text_processor import TextProcessor
from news_summarizer.utils.export import ExportManager


class NewsSummarizerGUI:
    """Windows 11 style news summarizer GUI."""

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("AI News Summarizer")
        self.root.geometry("1200x800")

        # Initialize components
        self.extractive_summarizer = ExtractiveSummarizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_clusterer = TopicClusterer()
        self.news_fetcher = NewsFetcher()
        self.text_processor = TextProcessor()
        self.export_manager = ExportManager()

        # Lazy load abstractive summarizer
        self.abstractive_summarizer = None

        # Data storage
        self.current_articles = []
        self.current_results = {}

        # Configure style for Windows 11 look
        self.setup_styles()

        # Create UI
        self.create_widgets()

    def setup_styles(self):
        """Setup Windows 11-like styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Modern colors
        bg_color = '#f3f3f3'
        fg_color = '#202020'
        accent_color = '#0067C0'

        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=accent_color, foreground='white',
                       borderwidth=0, focuscolor='none')
        style.map('TButton', background=[('active', '#005a9e')])

        self.root.configure(bg=bg_color)

    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_fetch_tab()
        self.create_analyze_tab()
        self.create_results_tab()

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_fetch_tab(self):
        """Create news fetching tab."""
        fetch_frame = ttk.Frame(self.notebook)
        self.notebook.add(fetch_frame, text="Fetch News")

        # Title
        title_label = ttk.Label(fetch_frame, text="Fetch News Articles",
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=20)

        # Source selection
        source_frame = ttk.LabelFrame(fetch_frame, text="Select News Sources", padding=10)
        source_frame.pack(fill=tk.X, padx=20, pady=10)

        self.source_vars = {}
        feeds = self.news_fetcher.get_available_feeds()

        row, col = 0, 0
        for source_name in feeds.keys():
            var = tk.BooleanVar(value=True)
            self.source_vars[source_name] = var
            cb = ttk.Checkbutton(source_frame, text=source_name.upper(),
                                variable=var)
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Options
        options_frame = ttk.LabelFrame(fetch_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(options_frame, text="Max articles per source:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_articles_var = tk.IntVar(value=10)
        ttk.Spinbox(options_frame, from_=1, to=50, textvariable=self.max_articles_var,
                   width=10).grid(row=0, column=1, padx=5, pady=5)

        # Fetch button
        fetch_btn = ttk.Button(fetch_frame, text="Fetch News",
                              command=self.fetch_news)
        fetch_btn.pack(pady=20)

        # Results area
        results_frame = ttk.LabelFrame(fetch_frame, text="Fetched Articles", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fetch_results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15)
        self.fetch_results_text.pack(fill=tk.BOTH, expand=True)

    def create_analyze_tab(self):
        """Create analysis tab."""
        analyze_frame = ttk.Frame(self.notebook)
        self.notebook.add(analyze_frame, text="Analyze")

        # Title
        title_label = ttk.Label(analyze_frame, text="Analyze News Articles",
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=20)

        # Analysis options
        options_frame = ttk.LabelFrame(analyze_frame, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=20, pady=10)

        # Summarization
        sum_frame = ttk.Frame(options_frame)
        sum_frame.pack(fill=tk.X, pady=5)

        self.summarization_var = tk.StringVar(value="extractive")
        ttk.Radiobutton(sum_frame, text="Extractive Summarization",
                       variable=self.summarization_var,
                       value="extractive").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sum_frame, text="Abstractive Summarization",
                       variable=self.summarization_var,
                       value="abstractive").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sum_frame, text="Both",
                       variable=self.summarization_var,
                       value="both").pack(side=tk.LEFT, padx=10)

        # Summary length
        length_frame = ttk.Frame(options_frame)
        length_frame.pack(fill=tk.X, pady=5)

        ttk.Label(length_frame, text="Summary sentences:").pack(side=tk.LEFT, padx=10)
        self.summary_length_var = tk.IntVar(value=3)
        ttk.Spinbox(length_frame, from_=1, to=10,
                   textvariable=self.summary_length_var,
                   width=10).pack(side=tk.LEFT, padx=5)

        # Sentiment analysis
        self.sentiment_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Perform Sentiment Analysis",
                       variable=self.sentiment_var).pack(anchor=tk.W, pady=5, padx=10)

        # Topic clustering
        self.clustering_var = tk.BooleanVar(value=True)
        cluster_frame = ttk.Frame(options_frame)
        cluster_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(cluster_frame, text="Perform Topic Clustering",
                       variable=self.clustering_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(cluster_frame, text="Number of topics:").pack(side=tk.LEFT, padx=10)
        self.num_topics_var = tk.IntVar(value=5)
        ttk.Spinbox(cluster_frame, from_=2, to=20,
                   textvariable=self.num_topics_var,
                   width=10).pack(side=tk.LEFT, padx=5)

        # Analyze button
        analyze_btn = ttk.Button(analyze_frame, text="Analyze Articles",
                                command=self.analyze_articles)
        analyze_btn.pack(pady=20)

        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(analyze_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

        # Status
        self.analyze_status = ttk.Label(analyze_frame, text="")
        self.analyze_status.pack()

    def create_results_tab(self):
        """Create results display tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Title
        title_label = ttk.Label(results_frame, text="Analysis Results",
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=20)

        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(export_frame, text="Export to JSON",
                  command=lambda: self.export_results('json')).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to HTML",
                  command=lambda: self.export_results('html')).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to TXT",
                  command=lambda: self.export_results('txt')).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to Markdown",
                  command=lambda: self.export_results('md')).pack(side=tk.LEFT, padx=5)

        # Results display
        results_text_frame = ttk.Frame(results_frame)
        results_text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.results_text = scrolledtext.ScrolledText(
            results_text_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def fetch_news(self):
        """Fetch news from selected sources."""
        selected_sources = [source for source, var in self.source_vars.items()
                           if var.get()]

        if not selected_sources:
            messagebox.showwarning("No Sources", "Please select at least one news source")
            return

        self.status_bar.config(text="Fetching news...")
        self.fetch_results_text.delete(1.0, tk.END)

        def fetch_thread():
            max_articles = self.max_articles_var.get()
            result = self.news_fetcher.fetch_default_feeds(
                categories=selected_sources,
                max_entries_per_feed=max_articles
            )

            self.current_articles = result.get('articles', [])

            # Display results
            self.root.after(0, self.display_fetch_results, result)

        threading.Thread(target=fetch_thread, daemon=True).start()

    def display_fetch_results(self, result):
        """Display fetched news results."""
        self.fetch_results_text.delete(1.0, tk.END)

        if not result.get('success', False):
            self.fetch_results_text.insert(tk.END,
                f"Error fetching news: {result.get('error', 'Unknown error')}\n")
            self.status_bar.config(text="Error fetching news")
            return

        articles = result.get('articles', [])
        self.fetch_results_text.insert(tk.END,
            f"Fetched {len(articles)} articles from {result.get('successful_feeds', 0)} sources\n\n")

        for i, article in enumerate(articles[:20], 1):  # Show first 20
            self.fetch_results_text.insert(tk.END,
                f"{i}. {article.get('title', 'Untitled')}\n")
            self.fetch_results_text.insert(tk.END,
                f"   Source: {article.get('source', 'Unknown')} | "
                f"Published: {article.get('published', 'N/A')}\n\n")

        if len(articles) > 20:
            self.fetch_results_text.insert(tk.END,
                f"... and {len(articles) - 20} more articles\n")

        self.status_bar.config(text=f"Fetched {len(articles)} articles")

    def analyze_articles(self):
        """Analyze fetched articles."""
        if not self.current_articles:
            messagebox.showwarning("No Articles",
                "Please fetch news articles first")
            return

        self.analyze_status.config(text="Analyzing articles...")
        self.progress_var.set(0)

        def analyze_thread():
            try:
                results = {'articles': []}
                total = len(self.current_articles)

                for i, article in enumerate(self.current_articles):
                    article_result = {
                        'title': article.get('title', ''),
                        'source': article.get('source', ''),
                        'published': article.get('published', ''),
                        'link': article.get('link', '')
                    }

                    text = f"{article.get('title', '')} {article.get('description', '')}"

                    # Summarization
                    if self.summarization_var.get() in ['extractive', 'both']:
                        summary = self.extractive_summarizer.summarize(
                            text, num_sentences=self.summary_length_var.get())
                        article_result['extractive_summary'] = summary

                    if self.summarization_var.get() in ['abstractive', 'both']:
                        if not self.abstractive_summarizer:
                            self.root.after(0, self.analyze_status.config,
                                          {'text': 'Loading abstractive model...'})
                            self.abstractive_summarizer = AbstractiveSummarizer()

                        summary = self.abstractive_summarizer.summarize(text)
                        article_result['abstractive_summary'] = summary

                    # Sentiment analysis
                    if self.sentiment_var.get():
                        sentiment = self.sentiment_analyzer.analyze(text)
                        article_result['sentiment'] = sentiment

                    results['articles'].append(article_result)

                    # Update progress
                    progress = ((i + 1) / total) * 90
                    self.root.after(0, self.progress_var.set, progress)

                # Topic clustering
                if self.clustering_var.get() and len(self.current_articles) >= 2:
                    self.root.after(0, self.analyze_status.config,
                                  {'text': 'Clustering topics...'})

                    texts = [f"{a.get('title', '')} {a.get('description', '')}"
                            for a in self.current_articles]

                    clustering = self.topic_clusterer.lda_topics(
                        texts, n_topics=self.num_topics_var.get())
                    results['clustering'] = clustering

                self.root.after(0, self.progress_var.set, 100)
                self.current_results = results

                # Display results
                self.root.after(0, self.display_results, results)
                self.root.after(0, self.analyze_status.config,
                              {'text': 'Analysis complete!'})
                self.root.after(0, self.status_bar.config,
                              {'text': 'Analysis complete'})

            except Exception as e:
                self.root.after(0, messagebox.showerror, "Error",
                              f"Analysis error: {str(e)}")
                self.root.after(0, self.analyze_status.config,
                              {'text': f'Error: {str(e)}'})

        threading.Thread(target=analyze_thread, daemon=True).start()

    def display_results(self, results):
        """Display analysis results."""
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, "=" * 80 + "\n")
        self.results_text.insert(tk.END, "NEWS ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, "=" * 80 + "\n\n")

        for i, article in enumerate(results['articles'], 1):
            self.results_text.insert(tk.END, f"{i}. {article['title']}\n")
            self.results_text.insert(tk.END, "-" * 80 + "\n")
            self.results_text.insert(tk.END,
                f"Source: {article['source']} | Published: {article['published']}\n\n")

            # Show summaries
            if 'extractive_summary' in article:
                self.results_text.insert(tk.END, "Extractive Summary:\n")
                self.results_text.insert(tk.END,
                    f"{article['extractive_summary'].get('summary', '')}\n\n")

            if 'abstractive_summary' in article:
                self.results_text.insert(tk.END, "Abstractive Summary:\n")
                self.results_text.insert(tk.END,
                    f"{article['abstractive_summary'].get('summary', '')}\n\n")

            # Show sentiment
            if 'sentiment' in article:
                sent = article['sentiment']
                self.results_text.insert(tk.END,
                    f"Sentiment: {sent.get('sentiment', 'N/A').upper()} "
                    f"(Confidence: {sent.get('confidence', 0):.2f})\n\n")

            self.results_text.insert(tk.END, "\n")

        # Show topics
        if 'clustering' in results and results['clustering'].get('success'):
            self.results_text.insert(tk.END, "=" * 80 + "\n")
            self.results_text.insert(tk.END, "TOPIC ANALYSIS\n")
            self.results_text.insert(tk.END, "=" * 80 + "\n\n")

            topics = results['clustering'].get('topics', {})
            for topic_id, topic_info in topics.items():
                self.results_text.insert(tk.END, f"Topic {topic_id + 1}: ")
                self.results_text.insert(tk.END,
                    ", ".join(topic_info['terms'][:5]) + "\n")

        # Switch to results tab
        self.notebook.select(2)

    def export_results(self, format):
        """Export results to file."""
        if not self.current_results:
            messagebox.showwarning("No Results",
                "Please analyze articles first")
            return

        filetypes = {
            'json': [('JSON files', '*.json')],
            'html': [('HTML files', '*.html')],
            'txt': [('Text files', '*.txt')],
            'md': [('Markdown files', '*.md')]
        }

        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{format}",
            filetypes=filetypes.get(format, [('All files', '*.*')])
        )

        if filepath:
            success = self.export_manager.export(
                self.current_results, filepath, format)

            if success:
                messagebox.showinfo("Success",
                    f"Results exported to {filepath}")
            else:
                messagebox.showerror("Error",
                    "Failed to export results")


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    app = NewsSummarizerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
