"""
Paper trading module for simulating real-time trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading simulator for testing strategies in real-time.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        simulate_latency: bool = True,
        latency_ms: int = 100
    ):
        """
        Initialize paper trader.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
            simulate_latency: Whether to simulate network latency
            latency_ms: Simulated latency in milliseconds
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.simulate_latency = simulate_latency
        self.latency_ms = latency_ms

        self.reset()

    def reset(self) -> None:
        """Reset paper trading state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.portfolio_history = []
        self.start_time = datetime.now()

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Total portfolio value
        """
        position_value = 0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position['shares'] * current_prices[symbol]

        return self.cash + position_value

    def place_order(
        self,
        symbol: str,
        order_type: str,
        shares: int,
        price: float,
        order_id: str = None
    ) -> Dict:
        """
        Place a paper trading order.

        Args:
            symbol: Asset symbol
            order_type: 'BUY' or 'SELL'
            shares: Number of shares
            price: Limit price
            order_id: Optional order ID

        Returns:
            Order details
        """
        if self.simulate_latency:
            time.sleep(self.latency_ms / 1000.0)

        if order_id is None:
            order_id = f"ORDER_{len(self.orders) + 1}"

        order = {
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'shares': shares,
            'price': price,
            'status': 'PENDING',
            'timestamp': datetime.now()
        }

        self.orders.append(order)
        logger.info(f"Order placed: {order_type} {shares} shares of {symbol} at ${price}")

        return order

    def execute_order(self, order: Dict, current_price: float) -> bool:
        """
        Execute a pending order.

        Args:
            order: Order dictionary
            current_price: Current market price

        Returns:
            True if order executed successfully
        """
        if order['status'] != 'PENDING':
            return False

        symbol = order['symbol']
        order_type = order['type']
        shares = order['shares']

        # Apply slippage
        if order_type == 'BUY':
            execution_price = current_price * (1 + self.slippage)
        else:  # SELL
            execution_price = current_price * (1 - self.slippage)

        # Execute BUY order
        if order_type == 'BUY':
            cost = shares * execution_price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost

            if total_cost > self.cash:
                logger.warning(f"Insufficient funds to execute BUY order for {symbol}")
                order['status'] = 'REJECTED'
                return False

            self.cash -= total_cost

            if symbol in self.positions:
                # Add to existing position (average price)
                old_shares = self.positions[symbol]['shares']
                old_avg_price = self.positions[symbol]['avg_price']
                new_shares = old_shares + shares
                new_avg_price = (
                    (old_shares * old_avg_price + shares * execution_price) / new_shares
                )

                self.positions[symbol] = {
                    'shares': new_shares,
                    'avg_price': new_avg_price,
                    'entry_date': self.positions[symbol]['entry_date']
                }
            else:
                # New position
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': execution_price,
                    'entry_date': datetime.now()
                }

            self.trades.append({
                'order_id': order['order_id'],
                'symbol': symbol,
                'type': 'BUY',
                'shares': shares,
                'price': execution_price,
                'cost': total_cost,
                'timestamp': datetime.now()
            })

        # Execute SELL order
        elif order_type == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                logger.warning(f"Insufficient shares to execute SELL order for {symbol}")
                order['status'] = 'REJECTED'
                return False

            revenue = shares * execution_price
            commission_cost = revenue * self.commission
            total_revenue = revenue - commission_cost

            self.cash += total_revenue

            # Calculate profit/loss
            avg_price = self.positions[symbol]['avg_price']
            profit = (execution_price - avg_price) * shares
            profit_pct = (execution_price - avg_price) / avg_price * 100

            # Update position
            self.positions[symbol]['shares'] -= shares

            if self.positions[symbol]['shares'] == 0:
                del self.positions[symbol]

            self.trades.append({
                'order_id': order['order_id'],
                'symbol': symbol,
                'type': 'SELL',
                'shares': shares,
                'price': execution_price,
                'revenue': total_revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'timestamp': datetime.now()
            })

            logger.info(f"SELL executed: {shares} {symbol} @ ${execution_price:.2f} | "
                       f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")

        order['status'] = 'EXECUTED'
        order['execution_price'] = execution_price
        order['execution_time'] = datetime.now()

        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        for order in self.orders:
            if order['order_id'] == order_id and order['status'] == 'PENDING':
                order['status'] = 'CANCELLED'
                logger.info(f"Order {order_id} cancelled")
                return True

        return False

    def update_portfolio_history(self, current_prices: Dict[str, float]) -> None:
        """
        Update portfolio history.

        Args:
            current_prices: Current market prices
        """
        portfolio_value = self.get_portfolio_value(current_prices)

        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital
        })

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Position details or None
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict:
        """Get all current positions."""
        return self.positions.copy()

    def get_trade_history(self) -> List[Dict]:
        """Get all executed trades."""
        return self.trades.copy()

    def get_order_history(self) -> List[Dict]:
        """Get all orders."""
        return self.orders.copy()

    def get_performance_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get performance summary.

        Args:
            current_prices: Current market prices

        Returns:
            Performance summary dictionary
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital

        # Calculate trade statistics
        completed_trades = [t for t in self.trades if 'profit' in t]
        winning_trades = [t for t in completed_trades if t['profit'] > 0]
        losing_trades = [t for t in completed_trades if t['profit'] <= 0]

        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0

        avg_profit = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0

        # Calculate elapsed time
        elapsed_time = datetime.now() - self.start_time

        return {
            'initial_capital': self.initial_capital,
            'current_value': portfolio_value,
            'cash': self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'open_positions': len(self.positions),
            'elapsed_time': str(elapsed_time),
            'start_time': self.start_time
        }

    def close_all_positions(self, current_prices: Dict[str, float]) -> None:
        """
        Close all open positions.

        Args:
            current_prices: Current market prices
        """
        positions_to_close = list(self.positions.keys())

        for symbol in positions_to_close:
            if symbol in current_prices:
                shares = self.positions[symbol]['shares']
                order = self.place_order(symbol, 'SELL', shares, current_prices[symbol])
                self.execute_order(order, current_prices[symbol])

        logger.info("All positions closed")
