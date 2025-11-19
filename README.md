# Autonomous Trading Bot Simulator

A comprehensive autonomous trading bot simulator built for Windows 11 with advanced features including multiple trading strategies, backtesting, risk management, paper trading, machine learning predictions, and real-time analytics.

## 💻 Two Ways to Use

### 🖥️ Windows GUI Application (Recommended for Windows 11)

Modern, production-ready Windows 11 application with:
- **Full GUI Interface** - No command line required
- **System Tray Integration** - Runs in background
- **Real-Time Updates** - Live dashboard and charts
- **Windows Notifications** - Native toast notifications
- **Easy Installation** - Professional Windows installer
- **One-Click Trading** - Start/stop with a button

**Quick Start:**
1. Download `TradingBotSimulator-Setup.exe`
2. Run installer
3. Launch from Start Menu
4. See [QUICK_START.md](QUICK_START.md) for 5-minute tutorial

**Or run from source:**
```bash
python windows_app.py
# Or use: run_windows_app.bat
```

📖 **Documentation:**
- [Quick Start Guide](QUICK_START.md) - Get running in 5 minutes
- [User Guide](USER_GUIDE.md) - Complete feature documentation
- [Build Guide](WINDOWS_BUILD_GUIDE.md) - Build your own installer

### 🖥️ Command Line Interface (Cross-Platform)

Traditional CLI for advanced users and automation:
```bash
python main.py backtest --symbol AAPL --strategy macd
```

See [CLI Usage](#🎯-quick-start) below for details.

---

## 🚀 Features

### 1. **Multiple Strategy Algorithms**
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **Bollinger Bands**: Volatility-based trading strategy

### 2. **Backtesting Engine**
- Historical data analysis
- Comprehensive performance metrics
- Trade-by-trade analysis
- Multiple timeframe support (1m, 5m, 15m, 1h, 1d)

### 3. **Risk Management System**
- Position sizing based on portfolio percentage
- Stop-loss and take-profit automation
- Maximum drawdown protection
- Risk/reward ratio validation
- Kelly Criterion position sizing

### 4. **Paper Trading Mode**
- Simulated real-time trading
- Latency simulation
- Commission and slippage modeling
- Order management system

### 5. **Real-Time Market Data Integration**
- Yahoo Finance integration
- Support for stocks, crypto, and forex
- Real-time price updates
- Historical data caching

### 6. **Portfolio Optimization**
- Maximum Sharpe Ratio
- Minimum Volatility
- Risk Parity
- Equal Weight
- Efficient Frontier calculation

### 7. **Machine Learning Price Prediction**
- LSTM neural networks
- Random Forest
- Gradient Boosting
- Feature engineering
- Model evaluation metrics

### 8. **Multi-Asset Support**
- Stocks (AAPL, MSFT, GOOGL, etc.)
- Cryptocurrencies (BTC-USD, ETH-USD)
- Forex pairs (EURUSD=X)
- Custom watchlists

### 9. **Performance Analytics Dashboard**
- Interactive web-based dashboard
- Real-time portfolio tracking
- Equity curves and drawdown charts
- Returns distribution analysis
- Strategy comparison tools

### 10. **Alert System**
- Desktop notifications (Windows 11 compatible)
- Trade execution alerts
- Signal generation notifications
- Risk breach warnings
- Daily performance summaries

## 📋 Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space

## 🔧 Installation

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd ai-medical-diagnosis
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Some packages like `ta-lib` may require additional setup on Windows:
- Download TA-Lib from: https://github.com/mrjbq7/ta-lib#windows
- Or use pandas-ta as an alternative (already included)

### Step 3: Configure Settings

Edit `config.yaml` to customize:
- Trading parameters (capital, commission)
- Risk management settings
- Strategy parameters
- Asset watchlists
- Alert preferences

## 🎯 Quick Start

### 1. Run a Backtest

Test a strategy on historical data:

```bash
python main.py backtest --symbol AAPL --strategy macd
```

**Options**:
- `--symbol`: Stock symbol (default: AAPL)
- `--strategy`: macd, rsi, or bollinger
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)

**Example**:
```bash
python main.py backtest --symbol TSLA --strategy rsi --start-date 2023-01-01 --end-date 2024-12-31
```

### 2. Compare Strategies

Compare all strategies on a single asset:

```bash
python main.py compare --symbol AAPL
```

### 3. Optimize Portfolio

Find optimal portfolio allocation:

```bash
python main.py optimize --symbols AAPL MSFT GOOGL AMZN --optimize-method max_sharpe
```

**Optimization methods**:
- `max_sharpe`: Maximum Sharpe ratio
- `min_volatility`: Minimum volatility
- `equal_weight`: Equal allocation
- `risk_parity`: Risk parity allocation

### 4. Train ML Model

Train a machine learning model for price prediction:

```bash
python main.py ml --symbol AAPL
```

The model will:
- Train on historical data
- Evaluate performance
- Generate price predictions
- Display accuracy metrics

### 5. Paper Trading

Simulate live trading without real money:

```bash
python main.py paper --symbol AAPL --strategy macd
```

**Multi-asset trading**:
```bash
python main.py paper --symbols AAPL MSFT GOOGL --strategy bollinger
```

Press `Ctrl+C` to stop trading and view summary.

### 6. Launch Dashboard

View analytics in your web browser:

```bash
python main.py dashboard
```

Open your browser to: http://127.0.0.1:8050

## 📊 Configuration

### Trading Parameters

```yaml
trading:
  initial_capital: 100000.0
  commission: 0.001  # 0.1%
  slippage: 0.0005   # 0.05%
```

### Risk Management

```yaml
risk_management:
  max_position_size: 0.2  # 20% per position
  max_portfolio_risk: 0.02  # 2% max loss per trade
  stop_loss_percent: 0.05  # 5% stop loss
  take_profit_percent: 0.15  # 15% take profit
  max_drawdown: 0.20  # 20% max drawdown
```

### Strategy Parameters

```yaml
strategies:
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9

  rsi:
    period: 14
    oversold: 30
    overbought: 70

  bollinger:
    period: 20
    std_dev: 2
```

## 📁 Project Structure

```
trading-bot-simulator/
├── src/
│   ├── config/           # Configuration management
│   ├── data/             # Data fetching and processing
│   ├── strategies/       # Trading strategies
│   ├── backtesting/      # Backtesting engine
│   ├── risk_management/  # Risk management system
│   ├── paper_trading/    # Paper trading simulator
│   ├── portfolio/        # Portfolio optimization
│   ├── ml/               # Machine learning models
│   ├── analytics/        # Dashboard and analytics
│   ├── alerts/           # Alert system
│   └── engine/           # Main trading engine
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── main.py              # Main entry point
└── README.md            # This file
```

## 🎓 Usage Examples

### Example 1: Complete Backtest Analysis

```bash
# Run backtest
python main.py backtest --symbol AAPL --strategy macd

# Compare all strategies
python main.py compare --symbol AAPL

# Optimize for best strategy parameters
python main.py optimize --symbols AAPL MSFT GOOGL
```

### Example 2: ML-Enhanced Trading

```bash
# Train ML model
python main.py ml --symbol AAPL

# Run backtest with optimized portfolio
python main.py optimize --symbols AAPL MSFT GOOGL --optimize-method max_sharpe

# Start paper trading
python main.py paper --symbols AAPL MSFT GOOGL --strategy macd
```

### Example 3: Real-Time Monitoring

```bash
# Start paper trading in one terminal
python main.py paper --symbol AAPL --strategy rsi

# Launch dashboard in another terminal
python main.py dashboard
```

## 📈 Performance Metrics

The bot calculates comprehensive metrics:

- **Returns**: Total, annualized, cumulative
- **Risk**: Volatility, max drawdown, Value at Risk
- **Ratios**: Sharpe ratio, risk/reward ratio
- **Trading**: Win rate, profit factor, trade distribution
- **ML**: R², RMSE, MAE, MAPE

## 🔔 Alert System

Alerts are sent for:
- Trade executions
- Trading signals
- Risk breaches
- System errors
- Daily summaries

On Windows 11, desktop notifications appear as native toast notifications.

## 🛡️ Risk Management Features

- **Position Sizing**: Automatic calculation based on risk parameters
- **Stop Loss**: Automatic stop-loss orders
- **Take Profit**: Automatic take-profit targets
- **Drawdown Protection**: Trading stops if max drawdown exceeded
- **Portfolio Limits**: Maximum position sizes enforced

## 🤖 Machine Learning Models

### Supported Models:
1. **LSTM**: Long Short-Term Memory neural networks
2. **Random Forest**: Ensemble learning method
3. **Gradient Boosting**: Boosted decision trees

### Features Used:
- Price data (OHLCV)
- Technical indicators
- Moving averages
- Volatility measures
- Volume indicators
- Lagged features

## 📊 Dashboard Features

The web dashboard includes:
- Real-time portfolio value
- Equity curve visualization
- Drawdown chart
- Returns distribution
- Trade analysis
- Strategy comparison
- Technical indicators chart
- Price predictions

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: TA-Lib installation fails

**Solution**: Use pandas-ta (already included) or install precompiled TA-Lib for Windows

### Issue: No data fetched

**Solution**:
- Check internet connection
- Verify symbol is correct
- Try different date range
- Check if market is open for real-time data

### Issue: Dashboard doesn't open

**Solution**:
- Check port 8050 is not in use
- Try different port in config.yaml
- Ensure dash packages are installed

## 🔮 Future Enhancements

- Live trading with broker API integration
- More technical indicators
- Advanced ML models (Transformers, Reinforcement Learning)
- Multi-timeframe analysis
- News sentiment analysis
- Social media sentiment integration
- Automated strategy optimization
- Cloud deployment support

## ⚠️ Disclaimer

**This is a simulator for educational and testing purposes only.**

- Past performance does not guarantee future results
- Trading involves risk of financial loss
- Always do your own research before trading
- Start with paper trading before risking real money
- The authors are not responsible for any financial losses

## 📝 License

This project is for educational purposes. Use at your own risk.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

## 🎉 Acknowledgments

Built with:
- Python
- pandas, numpy, scikit-learn
- TensorFlow/Keras
- yfinance
- Plotly & Dash
- And many other open-source libraries

---

**Happy Trading! 📈🚀**

Remember: Always test strategies thoroughly with backtesting and paper trading before considering real trading.
