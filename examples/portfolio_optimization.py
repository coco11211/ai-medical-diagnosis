"""
Example: Portfolio optimization with multiple stocks
"""
import sys
sys.path.append('..')

from src.engine import TradingEngine

def main():
    # Initialize trading engine
    print("Initializing trading engine...")
    engine = TradingEngine()

    # Define portfolio
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    print(f"\nOptimizing portfolio for {len(symbols)} stocks...")
    print(f"Symbols: {', '.join(symbols)}")

    # Optimize for maximum Sharpe ratio
    result = engine.optimize_portfolio(
        symbols=symbols,
        method='max_sharpe'
    )

    # Print results
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION - MAXIMUM SHARPE RATIO")
    print("="*70)
    print(f"Expected Annual Return: {result['expected_return']:.2%}")
    print(f"Annual Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    print("\nOptimal Allocation:")
    for symbol, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight:.2%}")
    print("="*70)

    # Compare different optimization methods
    print("\nComparing optimization methods...")

    methods = ['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity']

    for method in methods:
        result = engine.optimize_portfolio(symbols=symbols, method=method)
        print(f"\n{method.upper()}:")
        print(f"  Return: {result['expected_return']:.2%}, "
              f"Volatility: {result['volatility']:.2%}, "
              f"Sharpe: {result['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()
