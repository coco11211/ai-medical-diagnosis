"""
Simple example: Running a backtest on AAPL with MACD strategy
"""
import sys
sys.path.append('..')

from src.engine import TradingEngine

def main():
    # Initialize trading engine
    print("Initializing trading engine...")
    engine = TradingEngine()

    # Run backtest
    print("\nRunning backtest on AAPL with MACD strategy...")
    results = engine.run_backtest(
        symbol='AAPL',
        strategy_name='macd',
        start_date='2023-01-01',
        end_date='2024-12-31'
    )

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

    metrics = results['metrics']
    print(f"\nAnnualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
