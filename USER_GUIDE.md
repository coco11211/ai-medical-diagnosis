# Trading Bot Simulator - User Guide

Complete guide for using the Trading Bot Simulator Windows 11 application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Backtesting](#backtesting)
4. [Paper Trading](#paper-trading)
5. [Portfolio Optimization](#portfolio-optimization)
6. [Machine Learning](#machine-learning)
7. [Settings](#settings)
8. [Tips & Best Practices](#tips--best-practices)
9. [FAQ](#faq)

---

## Getting Started

### Installation

1. **Download the installer**
   - Get `TradingBotSimulator-Setup.exe` from the official download page

2. **Run the installer**
   - Double-click the installer
   - Follow the installation wizard
   - Choose installation location
   - Select whether to create desktop shortcut

3. **First launch**
   - Application will initialize on first run
   - May take 30-60 seconds to load ML models
   - Internet connection required for market data

### System Requirements

- **OS:** Windows 11 (Windows 10 also supported)
- **RAM:** 4 GB minimum (8 GB recommended)
- **Storage:** 500 MB free space
- **Internet:** Required for market data and updates
- **Display:** 1280x720 minimum resolution

### Main Window

The application has six main tabs:
1. **📊 Dashboard** - Overview and real-time monitoring
2. **📈 Backtesting** - Test strategies on historical data
3. **💰 Paper Trading** - Simulate live trading
4. **📁 Portfolio** - Portfolio optimization tools
5. **🤖 Machine Learning** - Price prediction models
6. **⚙️ Settings** - Configuration and preferences

---

## Dashboard Overview

The dashboard provides a real-time overview of your trading performance.

### Performance Summary

**Total Value**
- Shows current portfolio value
- Updates in real-time during paper trading
- Starts at initial capital ($100,000 default)

**Return**
- Displays total return as a percentage
- Green = profit, Red = loss
- Calculated as: (Current Value - Initial Capital) / Initial Capital

**Trades**
- Total number of trades executed
- Includes both buy and sell orders

### Performance Charts

Interactive charts showing:
- **Equity Curve:** Portfolio value over time
- **Drawdown:** Peak-to-trough decline
- **Returns Distribution:** Histogram of trade returns

### Recent Trades Table

Displays recent trade activity:
- **Time:** When trade was executed
- **Symbol:** Stock/crypto symbol
- **Action:** BUY or SELL
- **Quantity:** Number of shares/units
- **Price:** Execution price
- **Value:** Total transaction value
- **P&L:** Profit/Loss for the trade

---

## Backtesting

Test trading strategies on historical data to evaluate performance.

### Running a Backtest

1. **Select Symbol**
   - Enter stock symbol (e.g., AAPL, MSFT, TSLA)
   - Also supports crypto (BTC-USD, ETH-USD)
   - And forex (EURUSD=X)

2. **Choose Strategy**
   - **MACD:** Trend-following momentum indicator
   - **RSI:** Oversold/overbought momentum oscillator
   - **Bollinger Bands:** Volatility-based mean reversion

3. **Set Date Range**
   - Start Date: Beginning of test period
   - End Date: End of test period
   - Minimum: 3 months for meaningful results
   - Recommended: 1-2 years

4. **Click "Run Backtest"**
   - Processing may take 10-60 seconds
   - Progress indicator shows status
   - Results display when complete

### Understanding Results

**Performance Metrics:**

- **Initial Capital:** Starting amount ($100,000 default)
- **Final Value:** Ending portfolio value
- **Total Return:** Overall profit/loss percentage
- **Number of Trades:** Total trades executed

**Advanced Metrics:**

- **Annualized Return:** Return scaled to yearly basis
- **Volatility:** Risk measure (standard deviation of returns)
- **Sharpe Ratio:** Risk-adjusted return (higher is better)
  - < 1: Poor
  - 1-2: Good
  - \> 2: Excellent
- **Max Drawdown:** Largest peak-to-trough decline
- **Win Rate:** Percentage of profitable trades

### Interpreting Results

**Good Strategy Indicators:**
- ✅ Positive total return
- ✅ Sharpe ratio > 1
- ✅ Max drawdown < 20%
- ✅ Win rate > 50%
- ✅ Consistent performance across different periods

**Warning Signs:**
- ❌ Negative returns
- ❌ Very high volatility
- ❌ Large drawdowns
- ❌ Low win rate
- ❌ Too few trades (< 10)

### Tips for Backtesting

1. **Test Multiple Symbols**
   - Different strategies work better for different assets
   - Tech stocks may respond differently than blue chips

2. **Test Multiple Timeframes**
   - Bull markets (2020-2021)
   - Bear markets (2022)
   - Sideways markets

3. **Be Wary of Overfitting**
   - Great backtest results don't guarantee future performance
   - Past performance ≠ future results

4. **Compare Strategies**
   - Test all three strategies on same symbol
   - Choose the best performer

---

## Paper Trading

Simulate live trading without risking real money.

### Starting Paper Trading

1. **Enter Symbols**
   - Single: `AAPL`
   - Multiple: `AAPL,MSFT,GOOGL` (comma-separated)
   - Mix types: `AAPL,BTC-USD,EURUSD=X`

2. **Select Strategy**
   - Choose from MACD, RSI, or Bollinger Bands
   - Strategy will be applied to all symbols

3. **Click "Start Paper Trading"**
   - Trading begins immediately
   - Updates every 60 seconds (default)
   - Continues until you stop it

### Monitoring Paper Trading

**Trading Status Panel**
- Real-time status updates
- Trade executions logged
- Signals generated
- Errors or warnings

**Current Positions Table**
- Active positions
- Entry price
- Current price
- Unrealized P&L
- Updated in real-time

**Performance Updates**
- Dashboard updates every 5 seconds
- Shows current portfolio value
- Running P&L
- Number of trades

### Stopping Paper Trading

1. **Click "Stop Trading"**
   - All positions closed at market price
   - Final summary generated
   - Results saved

2. **View Summary**
   - Total return
   - Total trades
   - Win rate
   - Final portfolio value

### Paper Trading Tips

1. **Start with One Symbol**
   - Learn how strategy works
   - Easier to monitor

2. **Monitor for at Least 1 Hour**
   - See how strategy responds to market
   - Observe trade entries/exits

3. **Try Different Market Conditions**
   - Test during market hours
   - Test during high volatility
   - Test during low volatility

4. **Use Realistic Capital**
   - Don't use amounts you wouldn't trade in real life
   - Helps with psychological preparation

---

## Portfolio Optimization

Find optimal asset allocation for your portfolio.

### Running Optimization

1. **Enter Symbols**
   - Comma-separated list
   - Minimum 2 symbols
   - Recommended: 4-8 symbols
   - Example: `AAPL,MSFT,GOOGL,AMZN,TSLA`

2. **Select Optimization Method**

   **Maximum Sharpe Ratio**
   - Maximizes risk-adjusted returns
   - Best for growth-focused portfolios
   - Aggressive allocation

   **Minimum Volatility**
   - Minimizes portfolio risk
   - Best for conservative investors
   - Defensive allocation

   **Equal Weight**
   - Equal allocation to all assets
   - Simple, diversified approach
   - Benchmark for comparison

   **Risk Parity**
   - Equal risk contribution from each asset
   - Balanced approach
   - Considers correlations

3. **Click "Optimize Portfolio"**
   - Analysis takes 10-30 seconds
   - Results display when complete

### Understanding Results

**Metrics:**

- **Expected Return:** Projected annual return
- **Volatility:** Expected annual risk
- **Sharpe Ratio:** Risk-adjusted return

**Optimal Weights:**
- Percentage allocation for each symbol
- Total should equal 100%
- 0% means asset not included

### Example

For `AAPL,MSFT,GOOGL,AMZN`:

**Maximum Sharpe Ratio:**
```
AAPL:  35%
MSFT:  30%
GOOGL: 25%
AMZN:  10%
```

This means invest:
- $35,000 in AAPL (if you have $100,000)
- $30,000 in MSFT
- $25,000 in GOOGL
- $10,000 in AMZN

### Portfolio Optimization Tips

1. **Diversify Across Sectors**
   - Tech, Finance, Healthcare, Energy
   - Reduces sector-specific risk

2. **Include Different Asset Types**
   - Stocks, Bonds (if available), Gold
   - Better diversification

3. **Rebalance Periodically**
   - Quarterly or annually
   - Maintain target allocation

4. **Consider Correlations**
   - Highly correlated assets provide less diversification
   - Mix large-cap and small-cap

---

## Machine Learning

Use AI to predict future price movements.

### Training a Model

1. **Enter Symbol**
   - Stock symbol to predict
   - Example: `AAPL`

2. **Click "Train Model"**
   - Downloads historical data
   - Engineers features
   - Trains LSTM neural network
   - Takes 1-3 minutes

3. **Review Training Results**
   - Model type used
   - Number of samples
   - Performance metrics

### Understanding Model Performance

**Metrics:**

- **R² Score:** How well model fits data
  - 1.0 = perfect fit
  - 0.8-0.99 = excellent
  - 0.5-0.8 = good
  - < 0.5 = poor

- **RMSE:** Average prediction error
  - Lower is better
  - Measured in price units

- **MAE:** Mean absolute error
  - Average deviation from actual price
  - More interpretable than RMSE

- **MAPE:** Mean absolute percentage error
  - Error as percentage
  - < 5%: Excellent
  - 5-10%: Good
  - \> 10%: Fair

### Making Predictions

1. **Click "Predict"** (after training)
   - Generates next 5 period predictions
   - Based on recent data
   - Updates with new data

2. **Interpret Predictions**
   - Day 1: Most reliable
   - Day 5: Least reliable
   - Use as guidance, not certainty

### ML Trading Tips

1. **Don't Rely Solely on Predictions**
   - Use with technical analysis
   - Combine with fundamental analysis
   - Consider market conditions

2. **Retrain Regularly**
   - Market conditions change
   - Retrain weekly or monthly
   - More data = better predictions

3. **Validate Predictions**
   - Compare predictions with actual prices
   - Track accuracy over time
   - Adjust strategy accordingly

4. **Understand Limitations**
   - Cannot predict black swan events
   - Accuracy decreases over time
   - Past patterns may not repeat

---

## Settings

Customize application behavior and trading parameters.

### Trading Settings

**Initial Capital**
- Starting amount for backtesting and paper trading
- Default: $100,000
- Range: $1,000 - $10,000,000

**Commission**
- Trading fee per transaction
- Default: 0.1% (0.001)
- Typical range: 0% - 0.3%

### Risk Management

**Max Position Size**
- Maximum allocation per position
- Default: 20% (0.2)
- Conservative: 10%
- Aggressive: 30%

**Stop Loss**
- Automatic exit when loss exceeds percentage
- Default: 5% (0.05)
- Protects against large losses

**Take Profit**
- Automatic exit when profit reaches percentage
- Default: 15% (0.15)
- Locks in gains

### Notifications

Enable/disable alerts for:
- **Trade Executions:** When trades are executed
- **Trading Signals:** When strategies generate signals
- **Risk Warnings:** When risk limits approached

### Saving Settings

Click "Save Settings" to apply changes.
Settings persist across application restarts.

---

## Tips & Best Practices

### General Trading

1. **Start with Backtesting**
   - Understand strategy behavior
   - Validate performance
   - Build confidence

2. **Use Paper Trading Extensively**
   - Practice before real money
   - Test in different market conditions
   - Develop discipline

3. **Manage Risk**
   - Never risk more than you can afford to lose
   - Use stop losses
   - Diversify portfolio

4. **Stay Educated**
   - Learn technical analysis
   - Understand fundamental analysis
   - Follow market news

### Application Usage

1. **Regular Data Updates**
   - Close and reopen application daily
   - Ensures latest market data
   - Refreshes ML models

2. **Monitor System Resources**
   - Close other applications during intensive tasks
   - Backtesting and ML training are CPU-intensive
   - Ensure stable internet connection

3. **Backup Settings**
   - Export settings regularly
   - Located in: `%APPDATA%\Trading Bot Simulator\`
   - Restore if needed

4. **Review Logs**
   - Check logs for errors
   - Located in: `%APPDATA%\Trading Bot Simulator\logs\`
   - Helpful for troubleshooting

---

## FAQ

### General Questions

**Q: Is this for real trading?**
A: No, this is a simulator for educational purposes only. It does not connect to real brokers or execute real trades.

**Q: Do I need an internet connection?**
A: Yes, for downloading market data. Once data is cached, some features work offline.

**Q: What markets are supported?**
A: US stocks, cryptocurrencies, and major forex pairs via Yahoo Finance.

**Q: Is my data secure?**
A: All data is stored locally on your computer. No data is sent to external servers except market data requests.

### Trading Questions

**Q: Which strategy is best?**
A: It depends on the asset and market conditions. Use backtesting to compare strategies for your specific needs.

**Q: How often should I rebalance my portfolio?**
A: Quarterly or annually is common. More frequent rebalancing increases transaction costs.

**Q: Why are my backtest results different from paper trading?**
A: Backtesting uses historical data; paper trading uses real-time data. Market conditions change.

### Technical Questions

**Q: Application won't start**
A: Check logs in `%APPDATA%\Trading Bot Simulator\logs\`. Ensure Visual C++ Redistributable is installed.

**Q: Getting "No data" errors**
A: Check internet connection. Some symbols may not have data for selected date range.

**Q: Application is slow**
A: Close other applications. ML training and backtesting are resource-intensive. Ensure adequate RAM.

**Q: How do I update the application?**
A: Application checks for updates automatically. Or download latest installer from website.

### Data Questions

**Q: How often is data updated?**
A: Real-time data updates every 60 seconds during paper trading. Historical data is from Yahoo Finance.

**Q: Can I import my own data?**
A: Not in current version. Feature planned for future release.

**Q: Where is data stored?**
A: Cache: `%LOCALAPPDATA%\Trading Bot Simulator\cache\`
   Settings: `%APPDATA%\Trading Bot Simulator\`

---

## Support

### Getting Help

1. **Check this guide**
2. **Review logs** for error messages
3. **Visit GitHub** for known issues
4. **Contact support** via email

### Reporting Issues

When reporting issues, include:
- Windows version
- Application version
- Steps to reproduce
- Error messages
- Screenshots (if applicable)

### Community

- GitHub: https://github.com/yourusername/trading-bot-simulator
- Discord: [Link to Discord]
- Forum: [Link to Forum]

---

## Disclaimer

**IMPORTANT:** This software is for educational and simulation purposes only.

- ❌ Not financial advice
- ❌ Not a recommendation to trade
- ❌ Past performance ≠ future results
- ❌ Trading involves risk of loss

Always:
- ✅ Do your own research
- ✅ Understand the risks
- ✅ Consider consulting a licensed financial advisor
- ✅ Never invest more than you can afford to lose

---

## Appendix

### Keyboard Shortcuts

- `Ctrl+Q` - Quit application
- `Ctrl+S` - Save settings
- `F1` - Help
- `F5` - Refresh data

### Supported Symbols

**US Stocks:**
- AAPL, MSFT, GOOGL, AMZN, TSLA, etc.

**Cryptocurrencies:**
- BTC-USD, ETH-USD, ADA-USD, etc.

**Forex:**
- EURUSD=X, GBPUSD=X, USDJPY=X, etc.

### Strategy Parameters

Default parameters (can be customized in code):

**MACD:**
- Fast Period: 12
- Slow Period: 26
- Signal Period: 9

**RSI:**
- Period: 14
- Oversold: 30
- Overbought: 70

**Bollinger Bands:**
- Period: 20
- Standard Deviations: 2

---

**Happy Trading! 📈**

Version 1.0.0 | Last Updated: 2024
