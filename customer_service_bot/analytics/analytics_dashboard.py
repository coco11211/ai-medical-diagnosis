"""
Analytics Dashboard for Customer Service Bot
Real-time metrics and visualizations using Dash/Plotly
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationAnalytics:
    """
    Tracks and analyzes conversation metrics
    """

    def __init__(self):
        """Initialize analytics tracker"""
        self.conversations: List[Dict] = []
        self.metrics: Dict = {
            'total_conversations': 0,
            'total_messages': 0,
            'total_escalations': 0,
            'avg_sentiment': 0.0,
            'avg_conversation_length': 0.0,
            'resolution_rate': 0.0
        }

        # Time-series data
        self.hourly_stats = defaultdict(lambda: {
            'count': 0,
            'sentiment_sum': 0.0,
            'escalations': 0
        })

        # Intent tracking
        self.intent_counts = Counter()

        # Language tracking
        self.language_counts = Counter()

        logger.info("Conversation analytics initialized")

    def log_conversation(self, conversation_data: Dict):
        """
        Log a conversation for analytics

        Args:
            conversation_data: Dictionary with conversation metrics
        """
        self.conversations.append({
            **conversation_data,
            'timestamp': datetime.now()
        })

        # Update metrics
        self._update_metrics(conversation_data)

        # Update hourly stats
        hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
        self.hourly_stats[hour_key]['count'] += 1
        self.hourly_stats[hour_key]['sentiment_sum'] += conversation_data.get('avg_sentiment', 0)
        if conversation_data.get('escalated', False):
            self.hourly_stats[hour_key]['escalations'] += 1

        # Track intent
        intent = conversation_data.get('primary_intent', 'unknown')
        self.intent_counts[intent] += 1

        # Track language
        language = conversation_data.get('language', 'en')
        self.language_counts[language] += 1

        logger.debug(f"Logged conversation: {conversation_data.get('session_id', 'unknown')}")

    def _update_metrics(self, conversation_data: Dict):
        """Update aggregate metrics"""
        self.metrics['total_conversations'] += 1
        self.metrics['total_messages'] += conversation_data.get('message_count', 0)

        if conversation_data.get('escalated', False):
            self.metrics['total_escalations'] += 1

        # Update averages
        n = self.metrics['total_conversations']

        # Average sentiment
        old_avg_sent = self.metrics['avg_sentiment']
        new_sent = conversation_data.get('avg_sentiment', 0)
        self.metrics['avg_sentiment'] = old_avg_sent + (new_sent - old_avg_sent) / n

        # Average conversation length
        old_avg_len = self.metrics['avg_conversation_length']
        new_len = conversation_data.get('message_count', 0)
        self.metrics['avg_conversation_length'] = old_avg_len + (new_len - old_avg_len) / n

        # Resolution rate
        if conversation_data.get('resolved', False):
            resolved_count = sum(1 for c in self.conversations if c.get('resolved', False))
            self.metrics['resolution_rate'] = resolved_count / n

    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary"""
        return self.metrics.copy()

    def get_time_series_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get time-series data for the last N hours

        Args:
            hours: Number of hours to include

        Returns:
            DataFrame with hourly metrics
        """
        # Generate hour keys for the time range
        now = datetime.now()
        hour_keys = [
            (now - timedelta(hours=i)).strftime('%Y-%m-%d %H:00')
            for i in range(hours - 1, -1, -1)
        ]

        data = []
        for hour_key in hour_keys:
            stats = self.hourly_stats.get(hour_key, {
                'count': 0,
                'sentiment_sum': 0.0,
                'escalations': 0
            })

            avg_sentiment = stats['sentiment_sum'] / stats['count'] if stats['count'] > 0 else 0.0

            data.append({
                'hour': hour_key,
                'conversations': stats['count'],
                'avg_sentiment': avg_sentiment,
                'escalations': stats['escalations']
            })

        return pd.DataFrame(data)

    def get_intent_distribution(self) -> Dict:
        """Get intent distribution"""
        total = sum(self.intent_counts.values())
        if total == 0:
            return {}

        return {
            intent: count / total
            for intent, count in self.intent_counts.items()
        }

    def get_language_distribution(self) -> Dict:
        """Get language distribution"""
        total = sum(self.language_counts.values())
        if total == 0:
            return {}

        return {
            lang: count / total
            for lang, count in self.language_counts.items()
        }


class AnalyticsDashboard:
    """
    Dash-based analytics dashboard
    """

    def __init__(self, analytics: ConversationAnalytics, port: int = 8051):
        """
        Initialize dashboard

        Args:
            analytics: ConversationAnalytics instance
            port: Port to run dashboard on
        """
        self.analytics = analytics
        self.port = port

        # Create Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

        logger.info(f"Analytics dashboard initialized on port {port}")

    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Customer Service Bot Analytics Dashboard",
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5 * 1000,  # Update every 5 seconds
                n_intervals=0
            ),

            # Top metrics row
            html.Div([
                html.Div([
                    html.H3("Total Conversations"),
                    html.H2(id='total-conversations', style={'color': '#3498db'})
                ], className='metric-box', style={'flex': 1, 'padding': 20, 'margin': 10,
                                                   'backgroundColor': '#ecf0f1', 'borderRadius': 10}),

                html.Div([
                    html.H3("Total Messages"),
                    html.H2(id='total-messages', style={'color': '#2ecc71'})
                ], className='metric-box', style={'flex': 1, 'padding': 20, 'margin': 10,
                                                   'backgroundColor': '#ecf0f1', 'borderRadius': 10}),

                html.Div([
                    html.H3("Escalations"),
                    html.H2(id='total-escalations', style={'color': '#e74c3c'})
                ], className='metric-box', style={'flex': 1, 'padding': 20, 'margin': 10,
                                                   'backgroundColor': '#ecf0f1', 'borderRadius': 10}),

                html.Div([
                    html.H3("Avg Sentiment"),
                    html.H2(id='avg-sentiment', style={'color': '#9b59b6'})
                ], className='metric-box', style={'flex': 1, 'padding': 20, 'margin': 10,
                                                   'backgroundColor': '#ecf0f1', 'borderRadius': 10}),

            ], style={'display': 'flex', 'flexDirection': 'row'}),

            # Charts row 1
            html.Div([
                html.Div([
                    dcc.Graph(id='conversation-timeline')
                ], style={'flex': 1, 'padding': 10}),

                html.Div([
                    dcc.Graph(id='sentiment-timeline')
                ], style={'flex': 1, 'padding': 10}),
            ], style={'display': 'flex', 'flexDirection': 'row'}),

            # Charts row 2
            html.Div([
                html.Div([
                    dcc.Graph(id='intent-distribution')
                ], style={'flex': 1, 'padding': 10}),

                html.Div([
                    dcc.Graph(id='language-distribution')
                ], style={'flex': 1, 'padding': 10}),
            ], style={'display': 'flex', 'flexDirection': 'row'}),

            # Additional metrics
            html.Div([
                html.H3("Additional Metrics", style={'textAlign': 'center', 'marginTop': 20}),
                html.Div([
                    html.Div([
                        html.H4("Avg Conversation Length"),
                        html.H3(id='avg-conversation-length')
                    ], style={'flex': 1, 'textAlign': 'center'}),

                    html.Div([
                        html.H4("Resolution Rate"),
                        html.H3(id='resolution-rate')
                    ], style={'flex': 1, 'textAlign': 'center'}),

                    html.Div([
                        html.H4("Escalation Rate"),
                        html.H3(id='escalation-rate')
                    ], style={'flex': 1, 'textAlign': 'center'}),
                ], style={'display': 'flex', 'flexDirection': 'row', 'padding': 20})
            ])
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [
                Output('total-conversations', 'children'),
                Output('total-messages', 'children'),
                Output('total-escalations', 'children'),
                Output('avg-sentiment', 'children'),
                Output('avg-conversation-length', 'children'),
                Output('resolution-rate', 'children'),
                Output('escalation-rate', 'children'),
                Output('conversation-timeline', 'figure'),
                Output('sentiment-timeline', 'figure'),
                Output('intent-distribution', 'figure'),
                Output('language-distribution', 'figure'),
            ],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update all dashboard metrics"""
            metrics = self.analytics.get_metrics_summary()

            # Metric values
            total_conv = metrics['total_conversations']
            total_msg = metrics['total_messages']
            total_esc = metrics['total_escalations']
            avg_sent = f"{metrics['avg_sentiment']:.2f}"
            avg_len = f"{metrics['avg_conversation_length']:.1f} messages"
            res_rate = f"{metrics['resolution_rate'] * 100:.1f}%"
            esc_rate = f"{(total_esc / max(total_conv, 1)) * 100:.1f}%"

            # Time series data
            ts_data = self.analytics.get_time_series_data(hours=24)

            # Conversation timeline
            conv_timeline = go.Figure()
            conv_timeline.add_trace(go.Scatter(
                x=ts_data['hour'],
                y=ts_data['conversations'],
                mode='lines+markers',
                name='Conversations',
                line=dict(color='#3498db', width=2)
            ))
            conv_timeline.update_layout(
                title='Conversations Over Time (24h)',
                xaxis_title='Hour',
                yaxis_title='Count',
                hovermode='x unified'
            )

            # Sentiment timeline
            sent_timeline = go.Figure()
            sent_timeline.add_trace(go.Scatter(
                x=ts_data['hour'],
                y=ts_data['avg_sentiment'],
                mode='lines+markers',
                name='Avg Sentiment',
                line=dict(color='#9b59b6', width=2)
            ))
            sent_timeline.update_layout(
                title='Average Sentiment Over Time (24h)',
                xaxis_title='Hour',
                yaxis_title='Sentiment Score',
                hovermode='x unified'
            )

            # Intent distribution
            intent_dist = self.analytics.get_intent_distribution()
            if intent_dist:
                intent_fig = go.Figure(data=[go.Pie(
                    labels=list(intent_dist.keys()),
                    values=list(intent_dist.values()),
                    hole=0.3
                )])
                intent_fig.update_layout(title='Intent Distribution')
            else:
                intent_fig = go.Figure()
                intent_fig.update_layout(title='Intent Distribution (No Data)')

            # Language distribution
            lang_dist = self.analytics.get_language_distribution()
            if lang_dist:
                lang_fig = go.Figure(data=[go.Pie(
                    labels=list(lang_dist.keys()),
                    values=list(lang_dist.values()),
                    hole=0.3
                )])
                lang_fig.update_layout(title='Language Distribution')
            else:
                lang_fig = go.Figure()
                lang_fig.update_layout(title='Language Distribution (No Data)')

            return (
                total_conv, total_msg, total_esc, avg_sent,
                avg_len, res_rate, esc_rate,
                conv_timeline, sent_timeline, intent_fig, lang_fig
            )

    def run(self, debug: bool = False):
        """
        Run the dashboard

        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting analytics dashboard on http://127.0.0.1:{self.port}")
        print(f"\n{'=' * 70}")
        print(f"Analytics Dashboard Running")
        print(f"Open your browser to: http://127.0.0.1:{self.port}")
        print(f"{'=' * 70}\n")

        self.app.run_server(debug=debug, port=self.port, host='127.0.0.1')


# Testing function
def test_analytics_dashboard():
    """Test analytics dashboard with sample data"""
    import random
    from datetime import datetime, timedelta

    analytics = ConversationAnalytics()

    # Generate sample conversation data
    intents = ['account_issue', 'billing_payment', 'technical_support',
               'order_status', 'product_inquiry']
    languages = ['en', 'es', 'fr', 'de']

    print("Generating sample data...")

    for i in range(50):
        # Random timestamp within last 24 hours
        hours_ago = random.randint(0, 23)
        timestamp = datetime.now() - timedelta(hours=hours_ago)

        conversation_data = {
            'session_id': f'session_{i}',
            'message_count': random.randint(2, 15),
            'avg_sentiment': random.uniform(-1.0, 1.0),
            'primary_intent': random.choice(intents),
            'language': random.choice(languages),
            'escalated': random.random() < 0.2,  # 20% escalation rate
            'resolved': random.random() < 0.8,   # 80% resolution rate
            'duration_seconds': random.randint(60, 600),
        }

        analytics.log_conversation(conversation_data)

    print(f"Generated {analytics.metrics['total_conversations']} sample conversations")

    # Create and run dashboard
    dashboard = AnalyticsDashboard(analytics, port=8051)
    dashboard.run(debug=True)


if __name__ == "__main__":
    test_analytics_dashboard()
