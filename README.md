# Autonomous Trading Bot & Audio Noise Cancellation

A dual-purpose application built for Windows 11 featuring:

1. **Trading Bot Simulator**: Advanced autonomous trading with multiple strategies, backtesting, risk management, machine learning predictions, and real-time analytics
2. **Audio Noise Cancellation**: Real-time audio processing with low-latency noise removal optimized for Windows 11

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

---

## 🎤 Audio Noise Cancellation Features

### Real-Time Audio Processing
- **Live Noise Cancellation**: Process audio in real-time with minimal latency
- **Multiple Algorithms**:
  - Spectral Subtraction
  - Wiener Filtering
  - NoiseReduce Integration
- **Windows 11 Optimized**: WASAPI support for ultra-low latency
- **Adaptive Noise Profiling**: Learn and remove specific noise patterns
- **File Processing**: Batch process audio files with noise reduction
- **Live Monitoring**: Real-time audio visualization and statistics

### Audio Performance
- **Latency**: As low as 10-20ms (configurable)
- **Sample Rates**: 8kHz to 96kHz
- **Real-time Factor**: < 0.5x (processes faster than real-time)
- **Quality**: Professional-grade noise cancellation

### Use Cases
- Video calls and conferencing
- Live streaming and podcasting
- Gaming voice chat
- Content creation and recording
- Audio cleanup and restoration

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

### Trading Features

#### 1. Run a Backtest

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

---

### Audio Features

#### 1. List Audio Devices

Find your input/output devices:

```bash
python main.py audio-devices
```

#### 2. Real-Time Noise Cancellation

Start live audio processing:

```bash
# Basic usage with monitoring
python main.py audio-live --monitor

# With noise profile capture
python main.py audio-live --monitor --noise-profile

# Low latency for gaming
python main.py audio-live --monitor --latency-mode low

# High quality recording
python main.py audio-live --monitor --record --latency-mode high --noise-reduction 0.9
```

#### 3. Process Audio Files

Clean up existing audio files:

```bash
# Process a single file
python main.py audio-file --audio-input input.wav --audio-output cleaned.wav

# With noise profile from file
python main.py audio-file --audio-input input.wav --noise-profile --noise-reduction 0.8
```

**Audio Command Options**:
- `--monitor`: Enable real-time audio playback
- `--record`: Save processed audio to file
- `--noise-profile`: Capture noise profile before processing
- `--noise-reduction`: Strength (0.0 to 1.0, default: 0.8)
- `--latency-mode`: low/medium/high (default: low)
- `--visualize`: Enable real-time waveform visualization

**For detailed audio documentation**, see [AUDIO_GUIDE.md](AUDIO_GUIDE.md)

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
│   ├── engine/           # Main trading engine
│   ├── audio/            # Audio noise cancellation
│   │   ├── audio_capture.py      # Real-time audio capture
│   │   ├── noise_canceller.py    # Noise cancellation algorithms
│   │   ├── audio_processor.py    # Audio processing pipeline
│   │   ├── audio_config.py       # Audio configuration
│   │   ├── audio_utils.py        # Audio utilities
│   │   └── audio_visualizer.py   # Real-time visualization
│   └── examples/         # Usage examples
│       ├── audio_live_example.py
│       ├── audio_file_example.py
│       └── audio_config_example.py
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── main.py              # Main entry point
├── README.md            # This file
└── AUDIO_GUIDE.md       # Audio feature documentation
```

## 🎓 Usage Examples

### Trading Examples

#### Example 1: Complete Backtest Analysis

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

### Audio Examples

#### Example 1: Video Call Noise Reduction

```bash
# Capture ambient noise, then process live audio
python main.py audio-live --monitor --noise-profile --latency-mode low --noise-reduction 0.7
```

#### Example 2: Podcast Recording

```bash
# Record with noise cancellation
python main.py audio-live --record --latency-mode high --noise-reduction 0.9 --noise-profile
```

#### Example 3: Clean Existing Audio Files

```bash
# Process a recorded interview
python main.py audio-file --audio-input interview.wav --audio-output clean_interview.wav --noise-reduction 0.85
```

#### Example 4: Using Python API

```python
from src.audio import AudioProcessor, AudioConfig

# Create configuration
config = AudioConfig(
    latency_mode='low',
    noise_reduction_strength=0.8
)

# Create processor
processor = AudioProcessor(config)

# Capture noise profile
processor.capture_noise_profile(duration=2.0)

# Start real-time processing
processor.start_processing(monitor=True)
```

For more examples, see `src/examples/audio_*.py`


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

### Trading
- Live trading with broker API integration
- More technical indicators
- Advanced ML models (Transformers, Reinforcement Learning)
- Multi-timeframe analysis
- News sentiment analysis
- Social media sentiment integration
- Automated strategy optimization
- Cloud deployment support

### Audio
- RNNoise deep learning noise reduction
- Voice activity detection (VAD)
- Echo cancellation
- Automatic gain control (AGC)
- Multi-channel processing
- Real-time audio effects
- VST plugin support
- Integration with popular communication apps

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

**Trading**:
- Python, pandas, numpy, scikit-learn
- TensorFlow/Keras
- yfinance, backtrader
- Plotly & Dash

**Audio**:
- sounddevice, scipy
- noisereduce
- matplotlib
- numpy, librosa

---

**Happy Trading & Clear Audio! 📈🎤🚀**

**Trading**: Always test strategies thoroughly with backtesting and paper trading before considering real trading.

**Audio**: For comprehensive audio documentation, examples, and troubleshooting, see [AUDIO_GUIDE.md](AUDIO_GUIDE.md)
