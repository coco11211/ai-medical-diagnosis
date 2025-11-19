"""
Performance analytics dashboard using Plotly Dash.
"""
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

try:
    import dash
    from dash import dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Dash not available. Dashboard functionality will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dashboard:
    """
    Performance analytics dashboard for trading bot.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
        """
        Initialize dashboard.

        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.host = host
        self.port = port
        self.debug = debug

        if DASH_AVAILABLE:
            self.app = dash.Dash(
                __name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP]
            )
            self._setup_layout()
        else:
            self.app = None

    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Trading Bot Analytics Dashboard",
                           className="text-center mb-4 mt-4")
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Portfolio Value", className="card-title"),
                            html.H2(id="portfolio-value", children="$0.00")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Return", className="card-title"),
                            html.H2(id="total-return", children="0.00%")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Win Rate", className="card-title"),
                            html.H2(id="win-rate", children="0.00%")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2(id="sharpe-ratio", children="0.00")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="equity-curve")
                ], width=12)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="drawdown-chart")
                ], width=6),

                dbc.Col([
                    dcc.Graph(id="returns-distribution")
                ], width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="trade-analysis")
                ], width=6),

                dbc.Col([
                    dcc.Graph(id="strategy-comparison")
                ], width=6)
            ], className="mb-4"),

            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ], fluid=True)

    def create_equity_curve(self, portfolio_history: pd.DataFrame) -> go.Figure:
        """
        Create equity curve chart.

        Args:
            portfolio_history: DataFrame with portfolio history

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_history['date'] if 'date' in portfolio_history.columns else portfolio_history.index,
            y=portfolio_history['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86AB', width=2)
        ))

        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_drawdown_chart(self, portfolio_history: pd.DataFrame) -> go.Figure:
        """
        Create drawdown chart.

        Args:
            portfolio_history: DataFrame with portfolio history

        Returns:
            Plotly figure
        """
        # Calculate drawdown
        cumulative = portfolio_history['value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_history['date'] if 'date' in portfolio_history.columns else portfolio_history.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#A23B72', width=2)
        ))

        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_returns_distribution(self, returns: pd.Series) -> go.Figure:
        """
        Create returns distribution histogram.

        Args:
            returns: Series of returns

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#18A558'
        ))

        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_white'
        )

        return fig

    def create_trade_analysis(self, trades: pd.DataFrame) -> go.Figure:
        """
        Create trade analysis chart.

        Args:
            trades: DataFrame with trade history

        Returns:
            Plotly figure
        """
        if 'profit' not in trades.columns:
            # Create empty chart
            fig = go.Figure()
            fig.update_layout(title='Trade Analysis - No data available')
            return fig

        # Filter sell trades with profit information
        sell_trades = trades[trades['type'] == 'SELL'].copy()

        if sell_trades.empty:
            fig = go.Figure()
            fig.update_layout(title='Trade Analysis - No completed trades')
            return fig

        # Create bar chart of trade profits
        colors = ['green' if p > 0 else 'red' for p in sell_trades['profit']]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(len(sell_trades))),
            y=sell_trades['profit'],
            marker_color=colors,
            name='Trade Profit'
        ))

        fig.update_layout(
            title='Trade Profit/Loss Analysis',
            xaxis_title='Trade Number',
            yaxis_title='Profit/Loss ($)',
            template='plotly_white'
        )

        return fig

    def create_strategy_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create strategy comparison chart.

        Args:
            comparison_df: DataFrame comparing strategies

        Returns:
            Plotly figure
        """
        if comparison_df.empty:
            fig = go.Figure()
            fig.update_layout(title='Strategy Comparison - No data available')
            return fig

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=comparison_df['Strategy'],
            y=comparison_df['Total Return (%)'],
            name='Total Return',
            marker_color='#2E86AB'
        ))

        fig.update_layout(
            title='Strategy Performance Comparison',
            xaxis_title='Strategy',
            yaxis_title='Total Return (%)',
            template='plotly_white'
        )

        return fig

    def create_price_prediction_chart(
        self,
        historical: pd.DataFrame,
        predictions: np.ndarray
    ) -> go.Figure:
        """
        Create price prediction chart.

        Args:
            historical: Historical price data
            predictions: Predicted prices

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['close'],
            mode='lines',
            name='Historical',
            line=dict(color='#2E86AB', width=2)
        ))

        # Predictions
        if len(predictions) > 0:
            # Create future dates
            last_date = historical.index[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=len(predictions) + 1,
                freq='D'
            )[1:]

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#F18F01', width=2, dash='dash')
            ))

        fig.update_layout(
            title='Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_technical_indicators_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create technical indicators chart.

        Args:
            df: DataFrame with price and indicator data

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Bollinger Bands', 'MACD', 'RSI', 'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        # Price and Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)

        if 'bb_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='BB Upper', line=dict(color='gray', dash='dash')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='BB Lower', line=dict(color='gray', dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.2)'
            ), row=1, col=1)

        # MACD
        if 'macd' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD', line=dict(color='blue')
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd_signal'],
                name='Signal', line=dict(color='red')
            ), row=2, col=1)

        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI', line=dict(color='purple')
            ), row=3, col=1)

            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'],
            name='Volume', marker_color='lightblue'
        ), row=4, col=1)

        fig.update_layout(
            height=1000,
            title='Technical Indicators Analysis',
            showlegend=True,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )

        return fig

    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate text performance report.

        Args:
            results: Results dictionary from backtest

        Returns:
            Formatted report string
        """
        metrics = results.get('metrics', {})

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              TRADING BOT PERFORMANCE REPORT                  ║
╚══════════════════════════════════════════════════════════════╝

Strategy: {results.get('strategy', 'N/A')}

Portfolio Performance:
─────────────────────
Initial Capital:        ${results.get('initial_capital', 0):,.2f}
Final Value:            ${results.get('final_value', 0):,.2f}
Total Return:           {results.get('total_return_pct', 0):.2f}%
Annualized Return:      {metrics.get('annualized_return_pct', 0):.2f}%

Risk Metrics:
─────────────
Volatility:             {metrics.get('volatility_pct', 0):.2f}%
Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}
Maximum Drawdown:       {metrics.get('max_drawdown_pct', 0):.2f}%

Trading Statistics:
──────────────────
Total Trades:           {results.get('num_trades', 0)}
Winning Trades:         {metrics.get('winning_trades', 0)}
Losing Trades:          {metrics.get('losing_trades', 0)}
Win Rate:               {metrics.get('win_rate_pct', 0):.2f}%
Avg Trade Return:       {metrics.get('avg_trade_return_pct', 0):.2f}%

╚══════════════════════════════════════════════════════════════╝
        """

        return report

    def run(self) -> None:
        """Run the dashboard server."""
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Cannot run dashboard.")
            return

        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=self.debug)
