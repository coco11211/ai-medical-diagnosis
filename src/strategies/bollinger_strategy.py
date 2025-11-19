"""
Bollinger Bands trading strategy.
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal
import logging

logger = logging.getLogger(__name__)


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands trading strategy.

    Generates BUY signal when price touches/crosses below lower band.
    Generates SELL signal when price touches/crosses above upper band.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands strategy.

        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
        """
        params = {
            'period': period,
            'std_dev': std_dev
        }
        super().__init__('Bollinger Bands Strategy', params)
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bollinger Bands trading signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signal column
        """
        df = df.copy()

        # Ensure Bollinger Bands are calculated
        if 'bb_upper' not in df.columns:
            df['bb_middle'] = df['close'].rolling(window=self.period).mean()
            df['bb_std'] = df['close'].rolling(window=self.period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / df['bb_width']

        # Initialize signal column
        df['signal'] = 0

        # BUY when price crosses below lower band (oversold)
        df.loc[
            (df['close'] < df['bb_lower']) &
            (df['close'].shift(1) >= df['bb_lower'].shift(1)),
            'signal'
        ] = 1

        # SELL when price crosses above upper band (overbought)
        df.loc[
            (df['close'] > df['bb_upper']) &
            (df['close'].shift(1) <= df['bb_upper'].shift(1)),
            'signal'
        ] = -1

        # Mean reversion signals
        # BUY when price bounces off lower band
        df.loc[
            (df['close'] > df['bb_lower']) &
            (df['close'].shift(1) <= df['bb_lower'].shift(1)) &
            (df['low'] <= df['bb_lower']),
            'signal'
        ] = 1

        # SELL when price bounces off upper band
        df.loc[
            (df['close'] < df['bb_upper']) &
            (df['close'].shift(1) >= df['bb_upper'].shift(1)) &
            (df['high'] >= df['bb_upper']),
            'signal'
        ] = -1

        # Squeeze detection (low volatility -> breakout opportunity)
        df = self._detect_squeeze(df)

        logger.debug(f"Generated {(df['signal'] != 0).sum()} Bollinger Bands signals")

        return df

    def _detect_squeeze(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """
        Detect Bollinger Band squeeze (low volatility).

        Args:
            df: DataFrame with Bollinger Bands
            threshold: Squeeze threshold (% of price)

        Returns:
            DataFrame with squeeze detection
        """
        df['bb_squeeze'] = False

        # Normalize band width by price
        df['bb_width_normalized'] = df['bb_width'] / df['close']

        # Detect squeeze when band width is very narrow
        df.loc[
            df['bb_width_normalized'] < threshold,
            'bb_squeeze'
        ] = True

        # After squeeze, look for breakout
        for i in range(1, len(df)):
            if df['bb_squeeze'].iloc[i-1] and not df['bb_squeeze'].iloc[i]:
                # Breakout detected
                if df['close'].iloc[i] > df['bb_middle'].iloc[i]:
                    # Bullish breakout
                    df.loc[df.index[i], 'signal'] = 1
                elif df['close'].iloc[i] < df['bb_middle'].iloc[i]:
                    # Bearish breakout
                    df.loc[df.index[i], 'signal'] = -1

        return df

    def get_strength(self, df: pd.DataFrame, index: int = -1) -> float:
        """
        Get signal strength based on distance from bands.

        Args:
            df: DataFrame with Bollinger Bands data
            index: Index to check

        Returns:
            Signal strength (0-1)
        """
        if 'bb_percent' not in df.columns:
            return 0.0

        bb_percent = df['bb_percent'].iloc[index]

        # Strength based on how far price is from middle band
        # 0 = at lower band, 0.5 = at middle, 1 = at upper band
        strength = abs(bb_percent - 0.5) * 2

        return min(1.0, strength)

    def is_squeeze(self, df: pd.DataFrame, index: int = -1) -> bool:
        """
        Check if Bollinger Bands are in a squeeze.

        Args:
            df: DataFrame with Bollinger Bands
            index: Index to check

        Returns:
            True if in squeeze
        """
        if 'bb_squeeze' not in df.columns:
            return False

        return df['bb_squeeze'].iloc[index]
