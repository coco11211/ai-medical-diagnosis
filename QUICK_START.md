# Quick Start Guide - Windows 11 Application

Get up and running with the Trading Bot Simulator in 5 minutes!

## Installation (2 minutes)

### Option 1: Use the Installer (Recommended)

1. **Download** `TradingBotSimulator-Setup.exe`
2. **Run** the installer
3. **Click** through the installation wizard
4. **Launch** from Start Menu or Desktop shortcut

✅ Done! Skip to [First Run](#first-run).

### Option 2: Run from Source

1. **Install Python 3.8+** from https://python.org
2. **Download/Clone** this repository
3. **Open** PowerShell in the project directory
4. **Run:**
   ```powershell
   pip install -r requirements.txt
   python windows_app.py
   ```

## First Run (1 minute)

1. **Application loads** (may take 30-60 seconds first time)
2. **Main window appears** with 6 tabs
3. **You're ready!**

## Quick Tutorial (2 minutes)

### Test 1: Run a Backtest

Let's test Apple stock with a simple strategy:

1. Click **📈 Backtesting** tab
2. Symbol: `AAPL` (default)
3. Strategy: `MACD` (default)
4. Dates: Last 1 year (default)
5. Click **🚀 Run Backtest**
6. Wait 10-20 seconds
7. **Results appear!**

You should see:
- Total return percentage
- Number of trades
- Performance metrics
- Win rate

**Typical result:** 5-15% return over 1 year (varies by market conditions)

### Test 2: Try Paper Trading

Simulate real-time trading:

1. Click **💰 Paper Trading** tab
2. Symbols: `AAPL,MSFT`
3. Strategy: `RSI`
4. Click **▶️ Start Paper Trading**
5. Watch trades execute in real-time!
6. Click **⏹️ Stop Trading** after a minute
7. View summary

### Test 3: Optimize a Portfolio

Find the best allocation:

1. Click **📁 Portfolio** tab
2. Symbols: `AAPL,MSFT,GOOGL,AMZN`
3. Method: `Maximum Sharpe Ratio`
4. Click **🎯 Optimize Portfolio**
5. Wait 15-20 seconds
6. **See optimal weights!**

Example result:
```
AAPL:  35%
MSFT:  30%
GOOGL: 25%
AMZN:  10%
```

This tells you how to split your investment across these stocks.

## What's Next?

### Learn More

- **Full User Guide:** Open `USER_GUIDE.md`
- **Strategy Details:** Check the README
- **Build Guide:** See `WINDOWS_BUILD_GUIDE.md`

### Customize Settings

1. Go to **⚙️ Settings** tab
2. Adjust:
   - Initial capital
   - Risk parameters
   - Notifications
3. Click **💾 Save Settings**

### Try Advanced Features

**Machine Learning:**
1. **🤖 Machine Learning** tab
2. Enter symbol: `AAPL`
3. Click **🤖 Train Model**
4. Wait 2-3 minutes
5. Click **🔮 Predict**
6. See price predictions!

**Compare Strategies:**
Use backtesting to test all three strategies on the same symbol:
- MACD
- RSI
- Bollinger Bands

Pick the best performer!

## Common Issues

### "Module not found" error
**Solution:**
```powershell
pip install -r requirements.txt
```

### Application won't start
**Solutions:**
1. Install Visual C++ Redistributable
2. Check Python version: `python --version` (need 3.8+)
3. Check logs in: `%APPDATA%\Trading Bot Simulator\logs\`

### "No data" errors
**Solutions:**
1. Check internet connection
2. Try a different symbol
3. Try a different date range

### Windows SmartScreen warning
**Solution:**
Click "More info" → "Run anyway"
(This is normal for unsigned applications)

## Quick Commands

### Development Mode

```powershell
# Run GUI application
python windows_app.py

# Run CLI version - Backtest
python main.py backtest --symbol AAPL --strategy macd

# Run CLI version - Paper Trading
python main.py paper --symbol AAPL --strategy rsi

# Run CLI version - Portfolio Optimization
python main.py optimize --symbols AAPL MSFT GOOGL
```

### Build Executable

```powershell
# Build Windows executable
build_windows.bat

# Or manually
pyinstaller trading_bot.spec --clean
```

## Tips for Best Results

1. **Start with Backtesting**
   - Understand how strategies work
   - Test before paper trading

2. **Use Realistic Settings**
   - Default settings are good for most users
   - Adjust as you learn more

3. **Monitor Performance**
   - Check the Dashboard regularly
   - Track wins and losses

4. **Diversify**
   - Don't test just one stock
   - Try different sectors
   - Mix stocks, crypto, forex

5. **Be Patient**
   - Some operations take time
   - ML training: 2-3 minutes
   - Backtesting: 10-60 seconds
   - Portfolio optimization: 15-30 seconds

## System Tray Features

The app runs in the system tray:

- **Click tray icon** → Show/hide window
- **Right-click** → Menu
  - Show
  - Hide
  - Quit

Notifications appear for:
- Trade executions
- Trading signals
- Errors

## Data Storage

Your data is stored in:

**Settings:**
```
%APPDATA%\Trading Bot Simulator\settings.json
```

**Logs:**
```
%APPDATA%\Trading Bot Simulator\logs\
```

**Cache:**
```
%LOCALAPPDATA%\Trading Bot Simulator\cache\
```

## Getting Help

1. **User Guide:** `USER_GUIDE.md` (detailed instructions)
2. **README:** `README.md` (feature overview)
3. **Build Guide:** `WINDOWS_BUILD_GUIDE.md` (for developers)
4. **GitHub Issues:** Report bugs
5. **Logs:** Check for error details

## Safety Reminder

🚨 **IMPORTANT:** This is a simulator!

- ❌ No real money involved
- ❌ Not connected to real brokers
- ❌ For educational purposes only
- ✅ Practice before real trading
- ✅ Learn risk management
- ✅ Understand strategies

## Success Checklist

After Quick Start, you should have:

- ✅ Application installed and running
- ✅ Completed at least one backtest
- ✅ Tried paper trading
- ✅ Optimized a portfolio
- ✅ Explored all tabs
- ✅ Customized settings
- ✅ Understand basic features

## Next Steps

**Beginner:**
1. Read full User Guide
2. Test different strategies
3. Experiment with different stocks
4. Learn what metrics mean

**Intermediate:**
1. Try ML predictions
2. Compare strategies systematically
3. Optimize portfolios
4. Test in different market conditions

**Advanced:**
1. Read the code
2. Customize strategies
3. Add new indicators
4. Build your own features

## Resources

**Learning Materials:**
- [Investopedia](https://www.investopedia.com) - Trading basics
- [Python for Finance](https://www.oreilly.com/library/view/python-for-finance/9781492024330/) - Technical details
- [QuantConnect](https://www.quantconnect.com/docs) - Algorithmic trading

**Market Data:**
- [Yahoo Finance](https://finance.yahoo.com) - Stock data
- [TradingView](https://www.tradingview.com) - Charts
- [CoinMarketCap](https://coinmarketcap.com) - Crypto data

---

**You're all set! Start trading (virtually)! 🚀📈**

Remember: Practice makes perfect. Use this simulator to build your skills before risking real money.

---

Questions? Check the [FAQ in the User Guide](USER_GUIDE.md#faq)
