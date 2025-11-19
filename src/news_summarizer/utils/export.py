"""
Export Manager
Export summaries and analysis results to various formats.
"""

import json
from typing import Dict, List
from datetime import datetime
import os


class ExportManager:
    """Export news summaries and analysis to various formats."""

    def __init__(self):
        """Initialize export manager."""
        pass

    def export_to_json(self, data: Dict, filepath: str) -> bool:
        """
        Export data to JSON file.

        Args:
            data: Data to export
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def export_to_html(self, data: Dict, filepath: str) -> bool:
        """
        Export data to HTML file.

        Args:
            data: Data to export (should contain articles and summaries)
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            html = self._generate_html(data)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            return True
        except Exception as e:
            print(f"Error exporting to HTML: {e}")
            return False

    def _generate_html(self, data: Dict) -> str:
        """Generate HTML from data."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Summary Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .article {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .article h2 {{
            color: #333;
            margin-top: 0;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin: 10px 0;
        }}
        .summary {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
        }}
        .sentiment {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .sentiment.positive {{
            background: #d4edda;
            color: #155724;
        }}
        .sentiment.negative {{
            background: #f8d7da;
            color: #721c24;
        }}
        .sentiment.neutral {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        .topics {{
            margin: 15px 0;
        }}
        .topic-tag {{
            display: inline-block;
            background: #e7e7e7;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 5px 5px 5px 0;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>News Summary Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
"""

        # Add articles
        articles = data.get('articles', [])
        for i, article in enumerate(articles, 1):
            sentiment = article.get('sentiment', {})
            summary = article.get('summary', {})
            topics = article.get('topics', [])

            html += f"""
    <div class="article">
        <h2>{i}. {article.get('title', 'Untitled')}</h2>
        <div class="meta">
            <strong>Source:</strong> {article.get('source', 'Unknown')} |
            <strong>Published:</strong> {article.get('published', 'N/A')}
        </div>
"""

            # Add sentiment if available
            if sentiment:
                sent_label = sentiment.get('sentiment', 'neutral')
                html += f"""
        <div class="sentiment {sent_label}">
            Sentiment: {sent_label.capitalize()}
            (Confidence: {sentiment.get('confidence', 0):.2f})
        </div>
"""

            # Add summary
            if summary:
                html += f"""
        <div class="summary">
            <strong>Summary:</strong><br>
            {summary.get('summary', '')}
        </div>
"""

            # Add topics
            if topics:
                html += """
        <div class="topics">
            <strong>Topics:</strong><br>
"""
                for topic in topics:
                    html += f'            <span class="topic-tag">{topic}</span>\n'
                html += "        </div>\n"

            # Add link
            if article.get('link'):
                html += f"""
        <div class="meta">
            <a href="{article['link']}" target="_blank">Read full article →</a>
        </div>
"""

            html += "    </div>\n"

        html += """
</body>
</html>
"""
        return html

    def export_to_txt(self, data: Dict, filepath: str) -> bool:
        """
        Export data to plain text file.

        Args:
            data: Data to export
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("NEWS SUMMARY REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                articles = data.get('articles', [])
                for i, article in enumerate(articles, 1):
                    f.write(f"{i}. {article.get('title', 'Untitled')}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Source: {article.get('source', 'Unknown')}\n")
                    f.write(f"Published: {article.get('published', 'N/A')}\n")

                    sentiment = article.get('sentiment', {})
                    if sentiment:
                        f.write(f"Sentiment: {sentiment.get('sentiment', 'N/A')} ")
                        f.write(f"(Confidence: {sentiment.get('confidence', 0):.2f})\n")

                    summary = article.get('summary', {})
                    if summary:
                        f.write(f"\nSummary:\n{summary.get('summary', '')}\n")

                    topics = article.get('topics', [])
                    if topics:
                        f.write(f"\nTopics: {', '.join(topics)}\n")

                    if article.get('link'):
                        f.write(f"\nLink: {article['link']}\n")

                    f.write("\n" + "=" * 80 + "\n\n")

            return True
        except Exception as e:
            print(f"Error exporting to TXT: {e}")
            return False

    def export_to_markdown(self, data: Dict, filepath: str) -> bool:
        """
        Export data to Markdown file.

        Args:
            data: Data to export
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# News Summary Report\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write("---\n\n")

                articles = data.get('articles', [])
                for i, article in enumerate(articles, 1):
                    f.write(f"## {i}. {article.get('title', 'Untitled')}\n\n")
                    f.write(f"**Source:** {article.get('source', 'Unknown')}  \n")
                    f.write(f"**Published:** {article.get('published', 'N/A')}  \n")

                    sentiment = article.get('sentiment', {})
                    if sentiment:
                        f.write(f"**Sentiment:** {sentiment.get('sentiment', 'N/A')} ")
                        f.write(f"(Confidence: {sentiment.get('confidence', 0):.2f})  \n")

                    f.write("\n")

                    summary = article.get('summary', {})
                    if summary:
                        f.write(f"### Summary\n\n")
                        f.write(f"{summary.get('summary', '')}\n\n")

                    topics = article.get('topics', [])
                    if topics:
                        f.write(f"**Topics:** {', '.join(topics)}  \n")

                    if article.get('link'):
                        f.write(f"\n[Read full article]({article['link']})\n")

                    f.write("\n---\n\n")

            return True
        except Exception as e:
            print(f"Error exporting to Markdown: {e}")
            return False

    def export(self, data: Dict, filepath: str, format: str = 'auto') -> bool:
        """
        Export data to file (auto-detect format from extension).

        Args:
            data: Data to export
            filepath: Output file path
            format: Export format ('json', 'html', 'txt', 'md', or 'auto')

        Returns:
            True if successful
        """
        # Auto-detect format from extension
        if format == 'auto':
            ext = os.path.splitext(filepath)[1].lower()
            format_map = {
                '.json': 'json',
                '.html': 'html',
                '.htm': 'html',
                '.txt': 'txt',
                '.md': 'markdown',
                '.markdown': 'markdown'
            }
            format = format_map.get(ext, 'json')

        # Export based on format
        if format == 'json':
            return self.export_to_json(data, filepath)
        elif format == 'html':
            return self.export_to_html(data, filepath)
        elif format == 'txt':
            return self.export_to_txt(data, filepath)
        elif format in ['md', 'markdown']:
            return self.export_to_markdown(data, filepath)
        else:
            print(f"Unknown format: {format}")
            return False
