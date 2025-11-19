"""
RSI (Relative Strength Index) trading strategy.
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal
import logging

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI trading strategy.

    Generates BUY signal when RSI crosses above oversold threshold.
    Generates SELL signal when RSI crosses below overbought threshold.
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Initialize RSI strategy.

        Args:
            period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        params = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought
        }
        super().__init__('RSI Strategy', params)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RSI trading signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signal column
        """
        df = df.copy()

        # Ensure RSI is calculated
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        # Initialize signal column
        df['signal'] = 0

        # BUY when RSI crosses above oversold level
        df.loc[
            (df['rsi'] > self.oversold) &
            (df['rsi'].shift(1) <= self.oversold),
            'signal'
        ] = 1

        # SELL when RSI crosses below overbought level
        df.loc[
            (df['rsi'] < self.overbought) &
            (df['rsi'].shift(1) >= self.overbought),
            'signal'
        ] = -1

        # Additional logic: Extreme oversold/overbought
        # Strong BUY when RSI is extremely oversold (< 20)
        df.loc[
            (df['rsi'] < 20) &
            (df['rsi'].shift(1) >= 20),
            'signal'
        ] = 1

        # Strong SELL when RSI is extremely overbought (> 80)
        df.loc[
            (df['rsi'] > 80) &
            (df['rsi'].shift(1) <= 80),
            'signal'
        ] = -1

        # Divergence detection (advanced)
        df = self._detect_divergence(df)

        logger.debug(f"Generated {(df['signal'] != 0).sum()} RSI signals")

        return df

    def _detect_divergence(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences.

        Args:
            df: DataFrame with price and RSI data
            window: Lookback window for divergence

        Returns:
            DataFrame with divergence signals
        """
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high

        for i in range(window, len(df)):
            price_window = df['close'].iloc[i-window:i+1]
            rsi_window = df['rsi'].iloc[i-window:i+1]

            # Bullish divergence
            if (price_window.iloc[-1] < price_window.iloc[0] and
                rsi_window.iloc[-1] > rsi_window.iloc[0] and
                df['rsi'].iloc[i] < self.oversold):
                df.loc[df.index[i], 'signal'] = 1

            # Bearish divergence
            if (price_window.iloc[-1] > price_window.iloc[0] and
                rsi_window.iloc[-1] < rsi_window.iloc[0] and
                df['rsi'].iloc[i] > self.overbought):
                df.loc[df.index[i], 'signal'] = -1

        return df

    def get_strength(self, df: pd.DataFrame, index: int = -1) -> float:
        """
        Get signal strength based on RSI distance from neutral (50).

        Args:
            df: DataFrame with RSI data
            index: Index to check

        Returns:
            Signal strength (0-1)
        """
        if 'rsi' not in df.columns:
            return 0.0

        rsi = df['rsi'].iloc[index]

        # Strength increases as RSI moves away from 50
        strength = abs(rsi - 50) / 50

        return min(1.0, strength)
