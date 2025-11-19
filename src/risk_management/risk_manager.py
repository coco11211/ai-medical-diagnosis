"""
Risk management system for trading bot.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system to control trading risk.
    """

    def __init__(
        self,
        max_position_size: float = 0.2,
        max_portfolio_risk: float = 0.02,
        stop_loss_percent: float = 0.05,
        take_profit_percent: float = 0.15,
        max_drawdown: float = 0.20
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size as fraction of portfolio (0.2 = 20%)
            max_portfolio_risk: Maximum risk per trade as fraction (0.02 = 2%)
            stop_loss_percent: Stop loss percentage (0.05 = 5%)
            take_profit_percent: Take profit percentage (0.15 = 15%)
            max_drawdown: Maximum allowed drawdown (0.20 = 20%)
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.max_drawdown = max_drawdown

        self.positions = {}
        self.peak_portfolio_value = 0
        self.current_drawdown = 0

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float = None
    ) -> int:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            capital: Available capital
            price: Current asset price
            volatility: Asset volatility (optional, for position sizing)

        Returns:
            Number of shares to trade
        """
        # Calculate position size based on max position size
        max_position_value = capital * self.max_position_size

        # If volatility is provided, adjust for risk
        if volatility is not None:
            # Higher volatility = smaller position
            volatility_adjustment = 1 / (1 + volatility)
            max_position_value *= volatility_adjustment

        shares = int(max_position_value / price)

        logger.debug(f"Calculated position size: {shares} shares at ${price}")

        return max(1, shares)

    def calculate_stop_loss(self, entry_price: float, position_type: str = 'LONG') -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'

        Returns:
            Stop loss price
        """
        if position_type == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_percent)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.stop_loss_percent)

        return stop_loss

    def calculate_take_profit(self, entry_price: float, position_type: str = 'LONG') -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'

        Returns:
            Take profit price
        """
        if position_type == 'LONG':
            take_profit = entry_price * (1 + self.take_profit_percent)
        else:  # SHORT
            take_profit = entry_price * (1 - self.take_profit_percent)

        return take_profit

    def check_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        position_type: str = 'LONG'
    ) -> bool:
        """
        Check if stop loss is triggered.

        Args:
            current_price: Current price
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'

        Returns:
            True if stop loss triggered
        """
        stop_loss = self.calculate_stop_loss(entry_price, position_type)

        if position_type == 'LONG':
            return current_price <= stop_loss
        else:  # SHORT
            return current_price >= stop_loss

    def check_take_profit(
        self,
        current_price: float,
        entry_price: float,
        position_type: str = 'LONG'
    ) -> bool:
        """
        Check if take profit is triggered.

        Args:
            current_price: Current price
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'

        Returns:
            True if take profit triggered
        """
        take_profit = self.calculate_take_profit(entry_price, position_type)

        if position_type == 'LONG':
            return current_price >= take_profit
        else:  # SHORT
            return current_price <= take_profit

    def update_drawdown(self, current_portfolio_value: float) -> None:
        """
        Update current drawdown.

        Args:
            current_portfolio_value: Current portfolio value
        """
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        if self.peak_portfolio_value > 0:
            self.current_drawdown = (
                (self.peak_portfolio_value - current_portfolio_value) /
                self.peak_portfolio_value
            )

    def check_max_drawdown(self) -> bool:
        """
        Check if maximum drawdown exceeded.

        Returns:
            True if max drawdown exceeded
        """
        return self.current_drawdown >= self.max_drawdown

    def can_open_position(
        self,
        capital: float,
        current_positions: int,
        max_positions: int = 5
    ) -> bool:
        """
        Check if new position can be opened.

        Args:
            capital: Available capital
            current_positions: Number of current open positions
            max_positions: Maximum allowed positions

        Returns:
            True if position can be opened
        """
        # Check if max drawdown exceeded
        if self.check_max_drawdown():
            logger.warning("Maximum drawdown exceeded, cannot open new position")
            return False

        # Check if max positions exceeded
        if current_positions >= max_positions:
            logger.warning("Maximum positions reached")
            return False

        # Check if sufficient capital
        if capital <= 0:
            logger.warning("Insufficient capital")
            return False

        return True

    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """
        Calculate risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Risk/reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return 0

        return reward / risk

    def validate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_risk_reward: float = 2.0
    ) -> bool:
        """
        Validate trade based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            min_risk_reward: Minimum required risk/reward ratio

        Returns:
            True if trade is valid
        """
        rr_ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)

        if rr_ratio < min_risk_reward:
            logger.warning(f"Risk/reward ratio {rr_ratio:.2f} below minimum {min_risk_reward}")
            return False

        return True

    def calculate_portfolio_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            VaR value
        """
        if returns.empty:
            return 0.0

        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Optimal position size fraction
        """
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss  # Win/loss ratio
        kelly = (b * win_rate - (1 - win_rate)) / b

        # Use fractional Kelly (e.g., 0.25 Kelly) for safety
        return max(0, min(0.25, kelly * 0.5))

    def get_risk_metrics(self, portfolio_value: float) -> Dict:
        """
        Get current risk metrics.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of risk metrics
        """
        self.update_drawdown(portfolio_value)

        return {
            'current_drawdown': self.current_drawdown,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown_limit': self.max_drawdown,
            'max_drawdown_limit_pct': self.max_drawdown * 100,
            'peak_portfolio_value': self.peak_portfolio_value,
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'stop_loss_percent': self.stop_loss_percent,
            'take_profit_percent': self.take_profit_percent
        }
