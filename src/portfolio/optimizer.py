"""
Portfolio optimization module using Modern Portfolio Theory.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: DataFrame with price data (columns = symbols)

        Returns:
            DataFrame with returns
        """
        return prices.pct_change().dropna()

    def calculate_portfolio_performance(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Portfolio weights
            returns: Returns DataFrame

        Returns:
            Tuple of (expected return, volatility, sharpe ratio)
        """
        # Expected return
        expected_return = np.sum(returns.mean() * weights) * 252  # Annualized

        # Volatility
        cov_matrix = returns.cov() * 252  # Annualized
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        return expected_return, volatility, sharpe_ratio

    def optimize_max_sharpe(self, returns: pd.DataFrame) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio.

        Args:
            returns: Returns DataFrame

        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns.columns)

        # Generate random portfolios
        n_portfolios = 10000
        results = np.zeros((3, n_portfolios))
        weights_record = []

        for i in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            # Calculate metrics
            ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe
            weights_record.append(weights)

        # Find portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]

        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': results[0, max_sharpe_idx],
            'volatility': results[1, max_sharpe_idx],
            'sharpe_ratio': results[2, max_sharpe_idx],
            'method': 'max_sharpe'
        }

    def optimize_min_volatility(self, returns: pd.DataFrame) -> Dict:
        """
        Optimize portfolio for minimum volatility.

        Args:
            returns: Returns DataFrame

        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns.columns)

        # Generate random portfolios
        n_portfolios = 10000
        results = np.zeros((3, n_portfolios))
        weights_record = []

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe
            weights_record.append(weights)

        # Find portfolio with minimum volatility
        min_vol_idx = np.argmin(results[1])
        optimal_weights = weights_record[min_vol_idx]

        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': results[0, min_vol_idx],
            'volatility': results[1, min_vol_idx],
            'sharpe_ratio': results[2, min_vol_idx],
            'method': 'min_volatility'
        }

    def optimize_equal_weight(self, returns: pd.DataFrame) -> Dict:
        """
        Create equal-weighted portfolio.

        Args:
            returns: Returns DataFrame

        Returns:
            Dictionary with equal weights and metrics
        """
        n_assets = len(returns.columns)
        weights = np.array([1.0 / n_assets] * n_assets)

        ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

        return {
            'weights': dict(zip(returns.columns, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'method': 'equal_weight'
        }

    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict:
        """
        Create risk parity portfolio (equal risk contribution).

        Args:
            returns: Returns DataFrame

        Returns:
            Dictionary with risk parity weights and metrics
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252

        # Inverse volatility weights (simplified risk parity)
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)

        ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

        return {
            'weights': dict(zip(returns.columns, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'method': 'risk_parity'
        }

    def calculate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.

        Args:
            returns: Returns DataFrame
            n_portfolios: Number of portfolios to generate

        Returns:
            DataFrame with frontier portfolios
        """
        n_assets = len(returns.columns)
        results = []

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

            results.append({
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            })

        frontier_df = pd.DataFrame(results)

        # Sort by volatility and filter for efficient portfolios
        frontier_df = frontier_df.sort_values('volatility')

        return frontier_df

    def optimize_target_return(
        self,
        returns: pd.DataFrame,
        target_return: float
    ) -> Dict:
        """
        Optimize portfolio for target return with minimum volatility.

        Args:
            returns: Returns DataFrame
            target_return: Target annual return

        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns.columns)
        n_portfolios = 10000

        best_vol = float('inf')
        best_weights = None
        best_metrics = None

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            ret, vol, sharpe = self.calculate_portfolio_performance(weights, returns)

            # Find portfolio closest to target return with minimum volatility
            if abs(ret - target_return) < 0.01 and vol < best_vol:
                best_vol = vol
                best_weights = weights
                best_metrics = (ret, vol, sharpe)

        if best_weights is None:
            logger.warning(f"Could not find portfolio with target return {target_return}")
            return self.optimize_min_volatility(returns)

        return {
            'weights': dict(zip(returns.columns, best_weights)),
            'expected_return': best_metrics[0],
            'volatility': best_metrics[1],
            'sharpe_ratio': best_metrics[2],
            'method': 'target_return'
        }

    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate rebalancing trades.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Minimum difference to trigger rebalancing

        Returns:
            Dictionary of symbols -> weight changes
        """
        rebalance_trades = {}

        all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current

            if abs(diff) >= threshold:
                rebalance_trades[symbol] = diff

        return rebalance_trades

    def compare_strategies(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different portfolio optimization strategies.

        Args:
            returns: Returns DataFrame

        Returns:
            DataFrame comparing strategies
        """
        strategies = []

        # Max Sharpe
        max_sharpe = self.optimize_max_sharpe(returns)
        strategies.append({
            'Strategy': 'Max Sharpe',
            'Expected Return (%)': max_sharpe['expected_return'] * 100,
            'Volatility (%)': max_sharpe['volatility'] * 100,
            'Sharpe Ratio': max_sharpe['sharpe_ratio']
        })

        # Min Volatility
        min_vol = self.optimize_min_volatility(returns)
        strategies.append({
            'Strategy': 'Min Volatility',
            'Expected Return (%)': min_vol['expected_return'] * 100,
            'Volatility (%)': min_vol['volatility'] * 100,
            'Sharpe Ratio': min_vol['sharpe_ratio']
        })

        # Equal Weight
        equal = self.optimize_equal_weight(returns)
        strategies.append({
            'Strategy': 'Equal Weight',
            'Expected Return (%)': equal['expected_return'] * 100,
            'Volatility (%)': equal['volatility'] * 100,
            'Sharpe Ratio': equal['sharpe_ratio']
        })

        # Risk Parity
        risk_parity = self.optimize_risk_parity(returns)
        strategies.append({
            'Strategy': 'Risk Parity',
            'Expected Return (%)': risk_parity['expected_return'] * 100,
            'Volatility (%)': risk_parity['volatility'] * 100,
            'Sharpe Ratio': risk_parity['sharpe_ratio']
        })

        return pd.DataFrame(strategies)
