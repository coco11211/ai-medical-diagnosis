"""
Base strategy class for all trading strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, List
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, params: Dict = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.positions = {}
        self.signals_history = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            df: DataFrame with market data and indicators

        Returns:
            DataFrame with signal column added
        """
        pass

    def get_signal(self, df: pd.DataFrame, index: int = -1) -> Signal:
        """
        Get signal at specific index.

        Args:
            df: DataFrame with signals
            index: Index to get signal from

        Returns:
            Signal enum value
        """
        if 'signal' not in df.columns:
            return Signal.HOLD

        signal_value = df['signal'].iloc[index]

        if signal_value > 0:
            return Signal.BUY
        elif signal_value < 0:
            return Signal.SELL
        else:
            return Signal.HOLD

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        risk_per_trade: float = 0.02
    ) -> int:
        """
        Calculate position size based on risk management.

        Args:
            capital: Available capital
            price: Current price
            risk_per_trade: Maximum risk per trade (0.02 = 2%)

        Returns:
            Number of shares/units to trade
        """
        max_risk_amount = capital * risk_per_trade
        position_value = capital * 0.2  # Max 20% per position
        shares = int(position_value / price)

        return max(1, shares)

    def validate_signal(self, df: pd.DataFrame, index: int) -> bool:
        """
        Validate trading signal.

        Args:
            df: DataFrame with data
            index: Current index

        Returns:
            True if signal is valid
        """
        # Basic validation - ensure we have enough data
        if index < 0 or index >= len(df):
            return False

        # Check for NaN values in critical columns
        if df.iloc[index].isna().any():
            return False

        return True

    def backtest_signals(self, df: pd.DataFrame) -> Dict:
        """
        Backtest the strategy signals.

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with backtest results
        """
        df = self.generate_signals(df)

        trades = []
        position = None

        for i in range(len(df)):
            if not self.validate_signal(df, i):
                continue

            signal = self.get_signal(df, i)

            if signal == Signal.BUY and position is None:
                position = {
                    'entry_price': df['close'].iloc[i],
                    'entry_date': df.index[i],
                    'type': 'LONG'
                }

            elif signal == Signal.SELL and position is not None:
                position['exit_price'] = df['close'].iloc[i]
                position['exit_date'] = df.index[i]
                position['return'] = (
                    (position['exit_price'] - position['entry_price']) /
                    position['entry_price']
                )
                trades.append(position)
                position = None

        return {
            'total_trades': len(trades),
            'trades': trades,
            'signals': df['signal'].tolist() if 'signal' in df.columns else []
        }

    def get_strategy_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'parameters': self.params,
            'description': self.__doc__
        }

    def reset(self) -> None:
        """Reset strategy state."""
        self.positions = {}
        self.signals_history = []
        logger.info(f"Strategy {self.name} reset")
