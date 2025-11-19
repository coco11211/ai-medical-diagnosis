"""
Main trading engine integrating all components.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import time
import logging

from ..config import Config
from ..data import DataFetcher, DataProcessor
from ..strategies import MACDStrategy, RSIStrategy, BollingerStrategy
from ..backtesting import BacktestEngine
from ..risk_management import RiskManager
from ..paper_trading import PaperTrader
from ..portfolio import PortfolioOptimizer
from ..ml import MLPredictor
from ..alerts import AlertManager
from ..analytics import Dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that orchestrates all components.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize trading engine.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)

        # Initialize components
        self.data_fetcher = DataFetcher(
            source=self.config.get('data_sources.primary', 'yfinance')
        )

        self.data_processor = DataProcessor()

        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get('trading.initial_capital', 100000),
            commission=self.config.get('trading.commission', 0.001),
            slippage=self.config.get('trading.slippage', 0.0005)
        )

        self.risk_manager = RiskManager(
            max_position_size=self.config.get('risk_management.max_position_size', 0.2),
            max_portfolio_risk=self.config.get('risk_management.max_portfolio_risk', 0.02),
            stop_loss_percent=self.config.get('risk_management.stop_loss_percent', 0.05),
            take_profit_percent=self.config.get('risk_management.take_profit_percent', 0.15),
            max_drawdown=self.config.get('risk_management.max_drawdown', 0.20)
        )

        self.paper_trader = PaperTrader(
            initial_capital=self.config.get('trading.initial_capital', 100000),
            commission=self.config.get('trading.commission', 0.001),
            slippage=self.config.get('trading.slippage', 0.0005),
            simulate_latency=self.config.get('paper_trading.simulate_latency', True),
            latency_ms=self.config.get('paper_trading.latency_ms', 100)
        )

        self.portfolio_optimizer = PortfolioOptimizer()

        self.ml_predictor = MLPredictor(
            model_type=self.config.get('ml.model_type', 'random_forest'),
            lookback_period=self.config.get('ml.lookback_period', 60),
            prediction_horizon=self.config.get('ml.prediction_horizon', 5)
        )

        self.alert_manager = AlertManager(
            enable_email=self.config.get('alerts.enable_email', False),
            enable_desktop=self.config.get('alerts.enable_desktop', True),
            alert_on_trade=self.config.get('alerts.alert_on_trade', True),
            alert_on_signal=self.config.get('alerts.alert_on_signal', True),
            alert_on_risk_breach=self.config.get('alerts.alert_on_risk_breach', True)
        )

        self.dashboard = Dashboard(
            host=self.config.get('dashboard.host', '127.0.0.1'),
            port=self.config.get('dashboard.port', 8050),
            debug=self.config.get('dashboard.debug', True)
        )

        # Initialize strategies
        self.strategies = self._initialize_strategies()

        # State
        self.mode = "BACKTEST"  # BACKTEST, PAPER, LIVE
        self.is_running = False

        logger.info("Trading engine initialized successfully")

    def _initialize_strategies(self) -> Dict:
        """Initialize trading strategies."""
        strategies = {}

        # MACD Strategy
        macd_config = self.config.get('strategies.macd', {})
        strategies['macd'] = MACDStrategy(
            fast=macd_config.get('fast_period', 12),
            slow=macd_config.get('slow_period', 26),
            signal=macd_config.get('signal_period', 9)
        )

        # RSI Strategy
        rsi_config = self.config.get('strategies.rsi', {})
        strategies['rsi'] = RSIStrategy(
            period=rsi_config.get('period', 14),
            oversold=rsi_config.get('oversold', 30),
            overbought=rsi_config.get('overbought', 70)
        )

        # Bollinger Bands Strategy
        bb_config = self.config.get('strategies.bollinger', {})
        strategies['bollinger'] = BollingerStrategy(
            period=bb_config.get('period', 20),
            std_dev=bb_config.get('std_dev', 2)
        )

        return strategies

    def run_backtest(
        self,
        symbol: str,
        strategy_name: str = 'macd',
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Run backtest for a strategy.

        Args:
            symbol: Asset symbol
            strategy_name: Name of strategy to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest for {symbol} with {strategy_name} strategy")

        # Get configuration dates if not provided
        if start_date is None:
            start_date = self.config.get('backtesting.start_date', '2020-01-01')

        if end_date is None:
            end_date = self.config.get('backtesting.end_date', '2024-12-31')

        # Fetch historical data
        df = self.data_fetcher.fetch_historical_data(
            symbol,
            start_date,
            end_date,
            interval=self.config.get('backtesting.data_interval', '1d')
        )

        # Process data and add indicators
        df = self.data_processor.add_technical_indicators(df)
        df = self.data_processor.calculate_returns(df)
        df = self.data_processor.clean_data(df)

        # Get strategy
        strategy = self.strategies.get(strategy_name)

        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not found")

        # Run backtest
        results = self.backtest_engine.run_backtest(df, strategy, symbol)

        # Generate alert
        self.alert_manager.send_alert(
            title=f"Backtest Complete: {symbol}",
            message=f"Total Return: {results['total_return_pct']:.2f}%",
            alert_type="INFO"
        )

        logger.info(f"Backtest complete. Total return: {results['total_return_pct']:.2f}%")

        return results

    def compare_strategies(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Compare all strategies on a symbol.

        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame comparing strategies
        """
        logger.info(f"Comparing strategies for {symbol}")

        if start_date is None:
            start_date = self.config.get('backtesting.start_date', '2020-01-01')

        if end_date is None:
            end_date = self.config.get('backtesting.end_date', '2024-12-31')

        # Fetch data
        df = self.data_fetcher.fetch_historical_data(symbol, start_date, end_date)
        df = self.data_processor.add_technical_indicators(df)
        df = self.data_processor.clean_data(df)

        # Compare strategies
        strategies_list = list(self.strategies.values())
        comparison = self.backtest_engine.compare_strategies(df, strategies_list, symbol)

        return comparison

    def optimize_portfolio(self, symbols: List[str], method: str = 'max_sharpe') -> Dict:
        """
        Optimize portfolio allocation.

        Args:
            symbols: List of symbols
            method: Optimization method (max_sharpe, min_volatility, equal_weight, risk_parity)

        Returns:
            Optimization results
        """
        logger.info(f"Optimizing portfolio for {len(symbols)} assets using {method}")

        # Fetch data for all symbols
        data = self.data_fetcher.fetch_multiple_symbols(
            symbols,
            self.config.get('backtesting.start_date', '2020-01-01'),
            self.config.get('backtesting.end_date', '2024-12-31')
        )

        # Create price DataFrame
        prices = pd.DataFrame({
            symbol: data[symbol]['close']
            for symbol in symbols if symbol in data
        })

        # Calculate returns
        returns = self.portfolio_optimizer.calculate_returns(prices)

        # Optimize based on method
        if method == 'max_sharpe':
            result = self.portfolio_optimizer.optimize_max_sharpe(returns)
        elif method == 'min_volatility':
            result = self.portfolio_optimizer.optimize_min_volatility(returns)
        elif method == 'equal_weight':
            result = self.portfolio_optimizer.optimize_equal_weight(returns)
        elif method == 'risk_parity':
            result = self.portfolio_optimizer.optimize_risk_parity(returns)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        logger.info(f"Portfolio optimized. Expected return: {result['expected_return']:.2%}")

        return result

    def train_ml_model(self, symbol: str) -> Dict:
        """
        Train ML price prediction model.

        Args:
            symbol: Asset symbol

        Returns:
            Training results
        """
        logger.info(f"Training ML model for {symbol}")

        # Fetch historical data
        df = self.data_fetcher.fetch_historical_data(
            symbol,
            self.config.get('backtesting.start_date', '2020-01-01'),
            self.config.get('backtesting.end_date', '2024-12-31')
        )

        # Train model
        results = self.ml_predictor.train(
            df,
            train_split=self.config.get('ml.train_test_split', 0.8)
        )

        logger.info(f"ML model trained. R2 Score: {results['metrics'].get('r2', 0):.3f}")

        return results

    def predict_price(self, symbol: str, steps: int = 5) -> np.ndarray:
        """
        Predict future prices.

        Args:
            symbol: Asset symbol
            steps: Number of steps to predict

        Returns:
            Array of predictions
        """
        if not self.ml_predictor.is_trained:
            logger.warning("ML model not trained. Training now...")
            self.train_ml_model(symbol)

        # Fetch recent data
        df = self.data_fetcher.fetch_historical_data(
            symbol,
            self.config.get('backtesting.start_date', '2020-01-01'),
            datetime.now().strftime('%Y-%m-%d')
        )

        # Make prediction
        predictions = self.ml_predictor.predict(df, steps)

        logger.info(f"Predicted prices for {symbol}: {predictions}")

        return predictions

    def start_paper_trading(self, symbols: List[str], strategy_name: str = 'macd') -> None:
        """
        Start paper trading mode.

        Args:
            symbols: List of symbols to trade
            strategy_name: Strategy to use
        """
        logger.info(f"Starting paper trading for {symbols} with {strategy_name} strategy")

        self.mode = "PAPER"
        self.is_running = True

        strategy = self.strategies.get(strategy_name)

        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not found")

        self.alert_manager.send_alert(
            title="Paper Trading Started",
            message=f"Trading {len(symbols)} symbols with {strategy_name}",
            alert_type="INFO"
        )

        # This is a simplified paper trading loop
        # In production, this would run continuously with proper threading
        iteration = 0
        max_iterations = 100  # Limit for demo

        while self.is_running and iteration < max_iterations:
            for symbol in symbols:
                try:
                    # Fetch real-time data
                    current_data = self.data_fetcher.fetch_realtime_data(symbol)

                    # Get historical data for indicators
                    df = self.data_fetcher.fetch_historical_data(
                        symbol,
                        (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d')
                    )

                    # Add indicators
                    df = self.data_processor.add_technical_indicators(df)

                    # Generate signals
                    df = strategy.generate_signals(df)

                    # Get latest signal
                    signal = strategy.get_signal(df, -1)

                    # Execute trades based on signals
                    current_price = current_data['price']
                    position = self.paper_trader.get_position(symbol)

                    if signal.name == 'BUY' and position is None:
                        # Calculate position size
                        shares = self.risk_manager.calculate_position_size(
                            self.paper_trader.cash,
                            current_price
                        )

                        # Place order
                        order = self.paper_trader.place_order(
                            symbol, 'BUY', shares, current_price
                        )

                        self.paper_trader.execute_order(order, current_price)

                        self.alert_manager.alert_trade_executed(
                            symbol, 'BUY', shares, current_price
                        )

                    elif signal.name == 'SELL' and position is not None:
                        shares = position['shares']

                        order = self.paper_trader.place_order(
                            symbol, 'SELL', shares, current_price
                        )

                        self.paper_trader.execute_order(order, current_price)

                        # Calculate profit
                        profit = (current_price - position['avg_price']) * shares

                        self.alert_manager.alert_trade_executed(
                            symbol, 'SELL', shares, current_price, profit
                        )

                    # Update portfolio history
                    current_prices = {symbol: current_price}
                    self.paper_trader.update_portfolio_history(current_prices)

                except Exception as e:
                    logger.error(f"Error in paper trading loop for {symbol}: {e}")
                    self.alert_manager.alert_error(f"Paper trading error: {symbol}", str(e))

            iteration += 1
            time.sleep(60)  # Wait 1 minute between iterations

        logger.info("Paper trading stopped")

    def stop_trading(self) -> None:
        """Stop trading."""
        self.is_running = False
        logger.info("Trading stopped")

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if self.mode == "PAPER":
            # Get current prices
            symbols = list(self.paper_trader.positions.keys())
            current_prices = {}

            for symbol in symbols:
                try:
                    data = self.data_fetcher.fetch_realtime_data(symbol)
                    current_prices[symbol] = data['price']
                except:
                    pass

            return self.paper_trader.get_performance_summary(current_prices)

        return {}

    def run_dashboard(self) -> None:
        """Launch analytics dashboard."""
        logger.info("Launching dashboard...")
        self.dashboard.run()
