"""
Autonomous Trading Bot Simulator - Main Entry Point

A comprehensive trading bot with:
- Multiple strategy algorithms (MACD, RSI, Bollinger Bands)
- Backtesting engine
- Risk management
- Paper trading
- Real-time market data
- Portfolio optimization
- Machine learning price prediction
- Multi-asset support
- Performance analytics dashboard
- Alert system

Full Windows 11 compatibility
"""
import argparse
import sys
import logging
from datetime import datetime

from src.engine import TradingEngine
from src.analytics import Dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for trading bot."""
    parser = argparse.ArgumentParser(
        description='Autonomous Trading Bot Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'mode',
        choices=['backtest', 'paper', 'optimize', 'ml', 'dashboard', 'compare'],
        help='Trading bot mode'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol (default: AAPL)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Multiple symbols for portfolio optimization'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['macd', 'rsi', 'bollinger'],
        default='macd',
        help='Trading strategy (default: macd)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )

    parser.add_argument(
        '--optimize-method',
        type=str,
        choices=['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity'],
        default='max_sharpe',
        help='Portfolio optimization method'
    )

    args = parser.parse_args()

    # Initialize trading engine
    logger.info("Initializing trading engine...")
    engine = TradingEngine(config_path=args.config)

    try:
        if args.mode == 'backtest':
            # Run backtest
            logger.info(f"Running backtest for {args.symbol} with {args.strategy} strategy")

            results = engine.run_backtest(
                symbol=args.symbol,
                strategy_name=args.strategy,
                start_date=args.start_date,
                end_date=args.end_date
            )

            # Print results
            print("\n" + "="*70)
            print("BACKTEST RESULTS")
            print("="*70)
            print(f"Symbol: {args.symbol}")
            print(f"Strategy: {args.strategy.upper()}")
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Number of Trades: {results['num_trades']}")

            if 'metrics' in results:
                metrics = results['metrics']
                print(f"\nPerformance Metrics:")
                print(f"  Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
                print(f"  Volatility: {metrics.get('volatility_pct', 0):.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                print(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
            print("="*70 + "\n")

            # Generate detailed report
            report = engine.dashboard.generate_performance_report(results)
            print(report)

        elif args.mode == 'compare':
            # Compare strategies
            logger.info(f"Comparing strategies for {args.symbol}")

            comparison = engine.compare_strategies(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date
            )

            print("\n" + "="*70)
            print("STRATEGY COMPARISON")
            print("="*70)
            print(comparison.to_string(index=False))
            print("="*70 + "\n")

        elif args.mode == 'optimize':
            # Portfolio optimization
            symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

            logger.info(f"Optimizing portfolio for {len(symbols)} symbols")

            result = engine.optimize_portfolio(
                symbols=symbols,
                method=args.optimize_method
            )

            print("\n" + "="*70)
            print(f"PORTFOLIO OPTIMIZATION - {args.optimize_method.upper()}")
            print("="*70)
            print(f"Expected Return: {result['expected_return']:.2%}")
            print(f"Volatility: {result['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"\nOptimal Weights:")

            for symbol, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {symbol}: {weight:.2%}")
            print("="*70 + "\n")

        elif args.mode == 'ml':
            # Machine learning prediction
            logger.info(f"Training ML model for {args.symbol}")

            # Train model
            train_results = engine.train_ml_model(args.symbol)

            print("\n" + "="*70)
            print("ML MODEL TRAINING RESULTS")
            print("="*70)
            print(f"Model Type: {train_results['model_type']}")
            print(f"Training Samples: {train_results['train_samples']}")
            print(f"Test Samples: {train_results['test_samples']}")

            metrics = train_results['metrics']
            print(f"\nModel Performance:")
            print(f"  R² Score: {metrics.get('r2', 0):.4f}")
            print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"  MAE: {metrics.get('mae', 0):.4f}")
            print(f"  MAPE: {metrics.get('mape', 0):.2f}%")

            # Make predictions
            predictions = engine.predict_price(args.symbol, steps=5)

            print(f"\nPrice Predictions (next 5 periods):")
            for i, pred in enumerate(predictions, 1):
                print(f"  Day {i}: ${pred:.2f}")
            print("="*70 + "\n")

        elif args.mode == 'paper':
            # Paper trading
            symbols = args.symbols or [args.symbol]

            logger.info(f"Starting paper trading for {symbols}")

            print("\n" + "="*70)
            print("PAPER TRADING MODE")
            print("="*70)
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Strategy: {args.strategy.upper()}")
            print("Press Ctrl+C to stop trading")
            print("="*70 + "\n")

            try:
                engine.start_paper_trading(symbols, args.strategy)
            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                engine.stop_trading()

            # Print summary
            summary = engine.get_performance_summary()

            print("\n" + "="*70)
            print("PAPER TRADING SUMMARY")
            print("="*70)
            print(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
            print(f"Current Value: ${summary.get('current_value', 0):,.2f}")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Total Trades: {summary.get('total_trades', 0)}")
            print(f"Win Rate: {summary.get('win_rate_pct', 0):.2f}%")
            print("="*70 + "\n")

        elif args.mode == 'dashboard':
            # Launch dashboard
            logger.info("Launching analytics dashboard...")

            print("\n" + "="*70)
            print("ANALYTICS DASHBOARD")
            print("="*70)
            print("Dashboard will open in your web browser")
            print("Default URL: http://127.0.0.1:8050")
            print("Press Ctrl+C to stop the dashboard")
            print("="*70 + "\n")

            engine.run_dashboard()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         AUTONOMOUS TRADING BOT SIMULATOR v1.0                        ║
║                                                                      ║
║  Advanced Trading Bot with ML, Risk Management & Portfolio Optimization║
║                  Full Windows 11 Compatibility                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    main()
