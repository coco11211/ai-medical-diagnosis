"""
MACD (Moving Average Convergence Divergence) trading strategy.
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal
import logging

logger = logging.getLogger(__name__)


class MACDStrategy(BaseStrategy):
    """
    MACD trading strategy.

    Generates BUY signal when MACD crosses above signal line.
    Generates SELL signal when MACD crosses below signal line.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD strategy.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        params = {
            'fast_period': fast,
            'slow_period': slow,
            'signal_period': signal
        }
        super().__init__('MACD Strategy', params)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MACD trading signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signal column
        """
        df = df.copy()

        # Ensure MACD indicators are calculated
        if 'macd' not in df.columns:
            exp1 = df['close'].ewm(span=self.fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=self.slow, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Initialize signal column
        df['signal'] = 0

        # Generate signals based on MACD crossovers
        # BUY when MACD crosses above signal line
        df.loc[
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
            'signal'
        ] = 1

        # SELL when MACD crosses below signal line
        df.loc[
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
            'signal'
        ] = -1

        # Additional confirmation: check histogram direction
        df['histogram_increasing'] = df['macd_histogram'] > df['macd_histogram'].shift(1)

        # Strengthen signals with histogram confirmation
        df.loc[
            (df['signal'] == 1) & (df['histogram_increasing'] == False),
            'signal'
        ] = 0

        logger.debug(f"Generated {(df['signal'] != 0).sum()} MACD signals")

        return df

    def get_strength(self, df: pd.DataFrame, index: int = -1) -> float:
        """
        Get signal strength (0-1).

        Args:
            df: DataFrame with MACD data
            index: Index to check

        Returns:
            Signal strength
        """
        if 'macd_histogram' not in df.columns:
            return 0.0

        histogram = df['macd_histogram'].iloc[index]
        max_histogram = df['macd_histogram'].abs().max()

        if max_histogram == 0:
            return 0.0

        return abs(histogram) / max_histogram
