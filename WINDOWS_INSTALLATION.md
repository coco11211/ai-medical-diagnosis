# Windows 11 Installation Guide
## Autonomous Trading Bot Simulator

This guide will walk you through installing and running the Autonomous Trading Bot Simulator on Windows 11.

## Prerequisites

### 1. Python Installation

**Download and Install Python 3.11 or Higher:**

1. Visit [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.11 or later for Windows
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH" during installation
5. Click "Install Now"

**Verify Installation:**

Open Command Prompt (Win + R, type `cmd`, press Enter) and run:

```cmd
python --version
```

You should see: `Python 3.11.x` or higher

### 2. Git Installation (Optional but Recommended)

1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Verify: `git --version`

## Installation Steps

### Step 1: Download the Project

**Option A: Using Git (Recommended)**

```cmd
git clone <repository-url>
cd ai-medical-diagnosis
```

**Option B: Download ZIP**

1. Download the project as ZIP
2. Extract to a folder (e.g., `C:\TradingBot`)
3. Open Command Prompt in that folder

### Step 2: Create Virtual Environment (Recommended)

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Step 3: Install Dependencies

```cmd
pip install --upgrade pip
pip install numpy pandas scikit-learn plotly dash pyyaml
pip install yfinance
```

**If yfinance installation fails**, install dependencies individually:

```cmd
pip install pandas numpy requests beautifulsoup4 lxml html5lib
pip install platformdirs protobuf websockets frozendict peewee
pip install yfinance
```

**Alternative: Install all at once**

```cmd
pip install -r requirements.txt
```

**Note**: If `ta-lib` fails to install (common on Windows), it's okay - the bot uses alternative implementations.

### Step 4: Verify Installation

```cmd
python -c "from src.data import DataFetcher, DataProcessor; print('✓ Installation successful!')"
```

## Quick Start

### 1. Run Your First Backtest

Test the MACD strategy on Apple stock:

```cmd
python main.py backtest --symbol AAPL --strategy macd
```

### 2. Compare All Strategies

```cmd
python main.py compare --symbol AAPL
```

### 3. Start Paper Trading

Simulate live trading without risking real money:

```cmd
python main.py paper --symbol AAPL --strategy rsi
```

Press `Ctrl+C` to stop and view results.

### 4. Launch Analytics Dashboard

View interactive charts and analytics:

```cmd
python main.py dashboard
```

Open your browser to: `http://127.0.0.1:8050`

### 5. Train ML Model

Train a machine learning model for price prediction:

```cmd
python main.py ml --symbol AAPL
```

### 6. Portfolio Optimization

Find optimal portfolio allocation:

```cmd
python main.py optimize --symbols AAPL MSFT GOOGL AMZN --optimize-method max_sharpe
```

## Configuration

Edit `config.yaml` to customize:

- Trading capital
- Commission and slippage
- Risk management parameters
- Strategy settings
- Alert preferences

## Common Issues and Solutions

### Issue 1: "Python is not recognized"

**Solution:**
- Reinstall Python and check "Add Python to PATH"
- Or manually add Python to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit "Path" in System Variables
  3. Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python311`

### Issue 2: "No module named 'pandas'"

**Solution:**
```cmd
pip install pandas numpy
```

### Issue 3: "yfinance not available" warning

**Solution:**
```cmd
pip install yfinance
```

If that fails:
```cmd
pip install beautifulsoup4 lxml requests
pip install --no-deps yfinance
```

### Issue 4: "ta-lib installation failed"

**Solution:**
- This is optional; the bot works without it
- To install on Windows:
  1. Download TA-Lib from: [https://github.com/mrjbq7/ta-lib#windows](https://github.com/mrjbq7/ta-lib#windows)
  2. Or continue without it - pandas-ta is used as alternative

### Issue 5: "Dashboard doesn't open"

**Solution:**
- Check if port 8050 is available
- Change port in `config.yaml`:
  ```yaml
  dashboard:
    port: 8051  # or another port
  ```

### Issue 6: Windows Defender / Firewall Blocks

**Solution:**
- Allow Python through Windows Firewall when prompted
- Or temporarily disable firewall for testing

### Issue 7: "Permission Denied" errors

**Solution:**
- Run Command Prompt as Administrator
- Or install packages with `--user` flag:
  ```cmd
  pip install --user package-name
  ```

## Windows 11 Specific Features

### Desktop Notifications

The bot sends native Windows 11 toast notifications for:
- Trade executions
- Trading signals
- Risk breaches
- Daily summaries

**Enable notifications:**
1. Settings > System > Notifications
2. Ensure "Get notifications from apps and senders" is ON
3. Find Python in the list and enable notifications

### Task Scheduler (Optional)

Automate trading bot runs:

1. Open Task Scheduler (Win + R, type `taskschd.msc`)
2. Create Basic Task
3. Set trigger (daily, weekly, etc.)
4. Action: Start a program
5. Program: `C:\Path\To\venv\Scripts\python.exe`
6. Arguments: `C:\Path\To\main.py paper --symbol AAPL --strategy macd`

## Performance Tips for Windows 11

1. **Disable Windows Indexing** for the project folder (improves I/O)
2. **Add Python to Windows Defender exclusions** (faster execution)
3. **Use SSD** for better performance with historical data
4. **Close unnecessary applications** when running ML models
5. **Increase virtual memory** if running large backtests:
   - Settings > System > About > Advanced system settings
   - Performance > Settings > Advanced > Virtual memory

## Example Workflows

### Workflow 1: Strategy Testing

```cmd
# Activate virtual environment
venv\Scripts\activate

# Test MACD strategy
python main.py backtest --symbol AAPL --strategy macd --start-date 2023-01-01

# Test RSI strategy
python main.py backtest --symbol AAPL --strategy rsi --start-date 2023-01-01

# Test Bollinger Bands
python main.py backtest --symbol AAPL --strategy bollinger --start-date 2023-01-01

# Compare all
python main.py compare --symbol AAPL
```

### Workflow 2: Portfolio Analysis

```cmd
# Activate virtual environment
venv\Scripts\activate

# Optimize portfolio
python main.py optimize --symbols AAPL MSFT GOOGL AMZN TSLA --optimize-method max_sharpe

# Train ML model for each stock
python main.py ml --symbol AAPL
python main.py ml --symbol MSFT
python main.py ml --symbol GOOGL

# Start paper trading with optimized portfolio
python main.py paper --symbols AAPL MSFT GOOGL AMZN TSLA --strategy macd
```

### Workflow 3: Real-Time Monitoring

**Terminal 1:**
```cmd
venv\Scripts\activate
python main.py paper --symbols AAPL MSFT GOOGL --strategy rsi
```

**Terminal 2:**
```cmd
venv\Scripts\activate
python main.py dashboard
```

Open browser to `http://127.0.0.1:8050` for live monitoring.

## Updating the Bot

```cmd
git pull origin main
pip install --upgrade -r requirements.txt
```

## Uninstallation

1. Deactivate virtual environment: `deactivate`
2. Delete project folder
3. (Optional) Uninstall Python from Windows Settings

## Getting Help

- Check `README.md` for feature documentation
- Review configuration in `config.yaml`
- Check logs in `trading_bot.log`
- Ensure all dependencies are installed: `pip list`

## Next Steps

1. ✅ Verify installation with a backtest
2. ✅ Customize `config.yaml` for your preferences
3. ✅ Test paper trading before considering real trading
4. ✅ Explore the dashboard for visual analytics
5. ✅ Train ML models for better predictions
6. ✅ Compare strategies to find what works best

---

## Windows 11 Compatibility

✅ **Fully compatible with Windows 11**
- Native notifications
- Windows Terminal support
- PowerShell compatible
- Works with Windows Defender
- Optimized for Windows 11 performance

**Happy Trading! 📈**

*Remember: This is a simulator. Always test thoroughly before considering real trading.*
