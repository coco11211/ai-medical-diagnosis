"""
Quick Start Demo for Autonomous Trading Bot Simulator
This script demonstrates all key features of the trading bot.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataFetcher, DataProcessor
import pandas as pd

def demo_data_fetching():
    """Demonstrate data fetching capabilities."""
    print("\n" + "="*70)
    print("DEMO 1: DATA FETCHING")
    print("="*70)

    print("\nNote: This demo requires yfinance to be installed.")
    print("If you see an error, install it with: pip install yfinance\n")

    # Initialize data fetcher
    fetcher = DataFetcher(source='yfinance')

    try:
        # Fetch historical data for Apple
        print("Fetching historical data for AAPL...")
        data = fetcher.fetch_historical_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='1d'
        )

        print(f"\n✓ Successfully fetched {len(data)} days of data")
        print("\nFirst 5 rows:")
        print(data.head())

        print("\nLast 5 rows:")
        print(data.tail())

        # Get latest price
        latest_price = fetcher.get_latest_price('AAPL')
        print(f"\n✓ Latest AAPL price: ${latest_price:.2f}")

        return data

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTo fix this, install yfinance:")
        print("  pip install yfinance")
        return None


def demo_data_processing(data):
    """Demonstrate data processing capabilities."""
    print("\n" + "="*70)
    print("DEMO 2: DATA PROCESSING & TECHNICAL INDICATORS")
    print("="*70)

    if data is None:
        print("\nSkipping (no data available)")
        return None

    # Initialize data processor
    processor = DataProcessor()

    # Add technical indicators
    print("\nAdding technical indicators...")
    data_with_indicators = processor.add_technical_indicators(data)

    print(f"\n✓ Added {data_with_indicators.shape[1]} features")
    print("\nAvailable indicators:")
    indicators = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                  'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower', 'atr']

    for ind in indicators:
        if ind in data_with_indicators.columns:
            latest_value = data_with_indicators[ind].iloc[-1]
            if pd.notna(latest_value):
                print(f"  • {ind.upper()}: {latest_value:.2f}")

    # Show recent MACD signals
    print("\nRecent MACD Analysis:")
    recent = data_with_indicators.tail(5)[['close', 'macd', 'macd_signal', 'macd_histogram']]
    print(recent.to_string())

    # Show RSI
    print("\nRecent RSI Analysis:")
    recent_rsi = data_with_indicators.tail(5)[['close', 'rsi']]
    print(recent_rsi.to_string())

    latest_rsi = data_with_indicators['rsi'].iloc[-1]
    if latest_rsi < 30:
        print(f"\n📊 RSI Signal: OVERSOLD ({latest_rsi:.2f}) - Potential BUY")
    elif latest_rsi > 70:
        print(f"\n📊 RSI Signal: OVERBOUGHT ({latest_rsi:.2f}) - Potential SELL")
    else:
        print(f"\n📊 RSI Signal: NEUTRAL ({latest_rsi:.2f})")

    return data_with_indicators


def demo_strategy_signals(data):
    """Demonstrate trading strategy signals."""
    print("\n" + "="*70)
    print("DEMO 3: TRADING STRATEGY SIGNALS")
    print("="*70)

    if data is None:
        print("\nSkipping (no data available)")
        return

    # Get latest values
    latest = data.iloc[-1]

    print("\n📈 Current Market Analysis:")
    print(f"  Price: ${latest['close']:.2f}")
    print(f"  SMA 20: ${latest['sma_20']:.2f}")
    print(f"  SMA 50: ${latest['sma_50']:.2f}")
    print(f"  RSI: {latest['rsi']:.2f}")
    print(f"  MACD: {latest['macd']:.4f}")
    print(f"  MACD Signal: {latest['macd_signal']:.4f}")

    # MACD Strategy
    print("\n🔍 MACD Strategy:")
    if latest['macd'] > latest['macd_signal']:
        print("  ✓ BULLISH - MACD above signal line (BUY signal)")
    else:
        print("  ✗ BEARISH - MACD below signal line (SELL signal)")

    # RSI Strategy
    print("\n🔍 RSI Strategy:")
    if latest['rsi'] < 30:
        print("  ✓ OVERSOLD - Strong BUY signal")
    elif latest['rsi'] > 70:
        print("  ✗ OVERBOUGHT - Strong SELL signal")
    elif latest['rsi'] < 45:
        print("  ≈ SLIGHTLY OVERSOLD - Moderate BUY signal")
    elif latest['rsi'] > 55:
        print("  ≈ SLIGHTLY OVERBOUGHT - Moderate SELL signal")
    else:
        print("  ≈ NEUTRAL - Hold position")

    # Bollinger Bands Strategy
    print("\n🔍 Bollinger Bands Strategy:")
    if latest['close'] < latest['bb_lower']:
        print("  ✓ Price below lower band (BUY signal)")
    elif latest['close'] > latest['bb_upper']:
        print("  ✗ Price above upper band (SELL signal)")
    else:
        print("  ≈ Price within bands (NEUTRAL)")

    # Trend Analysis
    print("\n📊 Trend Analysis:")
    if latest['sma_20'] > latest['sma_50']:
        print("  ✓ SHORT-TERM UPTREND (SMA 20 > SMA 50)")
    else:
        print("  ✗ SHORT-TERM DOWNTREND (SMA 20 < SMA 50)")


def demo_risk_management():
    """Demonstrate risk management calculations."""
    print("\n" + "="*70)
    print("DEMO 4: RISK MANAGEMENT")
    print("="*70)

    capital = 100000
    risk_per_trade = 0.02  # 2%
    position_size = 0.2  # 20% max position
    stop_loss = 0.05  # 5%
    take_profit = 0.15  # 15%

    print(f"\nPortfolio Capital: ${capital:,.2f}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Max Position Size: {position_size*100}%")
    print(f"Stop Loss: {stop_loss*100}%")
    print(f"Take Profit: {take_profit*100}%")

    # Calculate position size
    max_position_value = capital * position_size
    risk_amount = capital * risk_per_trade

    print(f"\n💰 Position Sizing:")
    print(f"  Max Position Value: ${max_position_value:,.2f}")
    print(f"  Max Risk Amount: ${risk_amount:,.2f}")

    # Example trade
    entry_price = 150.00
    shares = int(max_position_value / entry_price)
    actual_position = shares * entry_price

    print(f"\n📊 Example Trade (Entry: ${entry_price}):")
    print(f"  Shares to Buy: {shares}")
    print(f"  Position Value: ${actual_position:,.2f}")
    print(f"  Stop Loss Price: ${entry_price * (1 - stop_loss):.2f}")
    print(f"  Take Profit Price: ${entry_price * (1 + take_profit):.2f}")
    print(f"  Max Loss: ${actual_position * stop_loss:,.2f}")
    print(f"  Potential Profit: ${actual_position * take_profit:,.2f}")
    print(f"  Risk/Reward Ratio: 1:{take_profit/stop_loss:.2f}")


def demo_performance_metrics():
    """Demonstrate performance metrics calculations."""
    print("\n" + "="*70)
    print("DEMO 5: PERFORMANCE METRICS")
    print("="*70)

    # Example backtest results
    initial_capital = 100000
    final_value = 125000
    trades = 50
    winning_trades = 32
    losing_trades = 18

    total_return = ((final_value - initial_capital) / initial_capital) * 100
    win_rate = (winning_trades / trades) * 100

    print(f"\n📈 Backtest Results:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Number of Trades: {trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")

    # Risk-adjusted metrics
    annualized_return = total_return * 2  # Assuming 6-month period
    volatility = 18.5  # Example annual volatility
    sharpe_ratio = annualized_return / volatility

    print(f"\n📊 Risk-Adjusted Metrics:")
    print(f"  Annualized Return: {annualized_return:.2f}%")
    print(f"  Annual Volatility: {volatility:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")

    max_drawdown = -15.5
    print(f"  Max Drawdown: {max_drawdown:.2f}%")

    if sharpe_ratio > 2:
        print("\n  ✓ EXCELLENT strategy (Sharpe > 2)")
    elif sharpe_ratio > 1:
        print("\n  ✓ GOOD strategy (Sharpe > 1)")
    elif sharpe_ratio > 0:
        print("\n  ≈ ACCEPTABLE strategy (Sharpe > 0)")
    else:
        print("\n  ✗ POOR strategy (Sharpe < 0)")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         AUTONOMOUS TRADING BOT SIMULATOR - QUICK START DEMO          ║
║                                                                      ║
║  This demo showcases the key features of the trading bot simulator  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run demos
    data = demo_data_fetching()
    processed_data = demo_data_processing(data)
    demo_strategy_signals(processed_data)
    demo_risk_management()
    demo_performance_metrics()

    # Final notes
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("""
Next Steps:
  1. Run a real backtest: python main.py backtest --symbol AAPL --strategy macd
  2. Compare strategies: python main.py compare --symbol AAPL
  3. Try paper trading: python main.py paper --symbol AAPL --strategy rsi
  4. Launch dashboard: python main.py dashboard
  5. Train ML model: python main.py ml --symbol AAPL
  6. Optimize portfolio: python main.py optimize --symbols AAPL MSFT GOOGL

For more information, see README.md and WINDOWS_INSTALLATION.md
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
