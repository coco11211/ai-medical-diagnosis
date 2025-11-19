"""
Backtesting engine for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()

    def reset(self) -> None:
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.portfolio_value = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy,
        symbol: str = "ASSET"
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            strategy: Trading strategy instance
            symbol: Asset symbol

        Returns:
            Dictionary with backtest results
        """
        self.reset()

        # Generate signals
        df = strategy.generate_signals(df)

        # Track portfolio value over time
        cash = self.initial_capital
        shares = 0
        position_type = None

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]

            # Calculate current portfolio value
            portfolio_value = cash + (shares * current_price)
            self.portfolio_value.append({
                'date': df.index[i],
                'value': portfolio_value,
                'cash': cash,
                'shares': shares,
                'price': current_price
            })

            # Execute trades based on signals
            if signal == 1 and shares == 0:  # BUY signal
                # Calculate shares to buy
                shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash

                if shares_to_buy > 0:
                    # Apply commission and slippage
                    buy_price = current_price * (1 + self.slippage)
                    cost = shares_to_buy * buy_price
                    commission_cost = cost * self.commission

                    cash -= (cost + commission_cost)
                    shares = shares_to_buy
                    position_type = 'LONG'

                    self.trades.append({
                        'type': 'BUY',
                        'date': df.index[i],
                        'price': buy_price,
                        'shares': shares_to_buy,
                        'cost': cost + commission_cost,
                        'portfolio_value': portfolio_value
                    })

            elif signal == -1 and shares > 0:  # SELL signal
                # Apply commission and slippage
                sell_price = current_price * (1 - self.slippage)
                revenue = shares * sell_price
                commission_cost = revenue * self.commission

                cash += (revenue - commission_cost)

                self.trades.append({
                    'type': 'SELL',
                    'date': df.index[i],
                    'price': sell_price,
                    'shares': shares,
                    'revenue': revenue - commission_cost,
                    'portfolio_value': portfolio_value
                })

                shares = 0
                position_type = None

        # Close any open positions at the end
        if shares > 0:
            final_price = df['close'].iloc[-1]
            sell_price = final_price * (1 - self.slippage)
            revenue = shares * sell_price
            commission_cost = revenue * self.commission
            cash += (revenue - commission_cost)

            self.trades.append({
                'type': 'SELL (Close)',
                'date': df.index[-1],
                'price': sell_price,
                'shares': shares,
                'revenue': revenue - commission_cost,
                'portfolio_value': cash
            })

            shares = 0

        # Calculate final portfolio value
        final_value = cash + (shares * df['close'].iloc[-1])

        # Calculate performance metrics
        metrics = self._calculate_metrics(df, final_value)

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': (final_value - self.initial_capital) / self.initial_capital,
            'total_return_pct': ((final_value - self.initial_capital) / self.initial_capital) * 100,
            'trades': self.trades,
            'num_trades': len(self.trades),
            'portfolio_value': self.portfolio_value,
            'metrics': metrics,
            'strategy': strategy.name
        }

    def _calculate_metrics(self, df: pd.DataFrame, final_value: float) -> Dict:
        """
        Calculate performance metrics.

        Args:
            df: DataFrame with price data
            final_value: Final portfolio value

        Returns:
            Dictionary of metrics
        """
        # Calculate returns
        portfolio_df = pd.DataFrame(self.portfolio_value)

        if portfolio_df.empty:
            return {}

        portfolio_df.set_index('date', inplace=True)
        portfolio_df['returns'] = portfolio_df['value'].pct_change()

        # Total return
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = portfolio_df['returns'].std() * np.sqrt(252)  # Annualized

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = 0
        losing_trades = 0

        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]

                if sell_trade['revenue'] > buy_trade['cost']:
                    winning_trades += 1
                else:
                    losing_trades += 1

        total_completed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0

        # Average trade
        trade_returns = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                trade_return = (sell_trade['revenue'] - buy_trade['cost']) / buy_trade['cost']
                trade_returns.append(trade_return)

        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_return': avg_trade_return,
            'avg_trade_return_pct': avg_trade_return * 100
        }

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: List,
        symbol: str = "ASSET"
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            df: DataFrame with OHLCV data
            strategies: List of strategy instances
            symbol: Asset symbol

        Returns:
            DataFrame comparing strategy results
        """
        results = []

        for strategy in strategies:
            result = self.run_backtest(df, strategy, symbol)

            results.append({
                'Strategy': strategy.name,
                'Total Return (%)': result['total_return_pct'],
                'Annualized Return (%)': result['metrics'].get('annualized_return_pct', 0),
                'Sharpe Ratio': result['metrics'].get('sharpe_ratio', 0),
                'Max Drawdown (%)': result['metrics'].get('max_drawdown_pct', 0),
                'Win Rate (%)': result['metrics'].get('win_rate_pct', 0),
                'Number of Trades': result['num_trades'],
                'Final Value': result['final_value']
            })

        return pd.DataFrame(results)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve DataFrame."""
        return pd.DataFrame(self.portfolio_value)

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log DataFrame."""
        return pd.DataFrame(self.trades)
