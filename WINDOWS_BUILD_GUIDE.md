# Windows 11 Application - Build & Deployment Guide

Complete guide for building and deploying the Trading Bot Simulator as a production-ready Windows 11 application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Building the Application](#building-the-application)
4. [Testing the Application](#testing-the-application)
5. [Creating the Installer](#creating-the-installer)
6. [Distribution](#distribution)
7. [Troubleshooting](#troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Prerequisites

### Required Software

1. **Windows 11** (or Windows 10)
   - 64-bit version required
   - Administrator access for installation

2. **Python 3.8 or Higher**
   - Download from: https://www.python.org/downloads/
   - ✅ Add Python to PATH during installation
   - ✅ Install pip

3. **Microsoft Visual C++ Redistributable**
   - Required for some Python packages
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

4. **Inno Setup (Optional, for installer)**
   - Download from: https://jrsoftware.org/isinfo.php
   - Used to create professional Windows installer
   - Add to PATH for command-line access

### Verify Installation

Open PowerShell or Command Prompt and verify:

```powershell
# Check Python
python --version
# Should output: Python 3.8.x or higher

# Check pip
pip --version

# Check git (if cloning from repository)
git --version
```

---

## Development Setup

### 1. Clone or Download Repository

```bash
git clone https://github.com/yourusername/trading-bot-simulator.git
cd trading-bot-simulator
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your prompt
```

### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note on TA-Lib:**
If TA-Lib installation fails, the application will use pandas-ta as a fallback (already included).

For TA-Lib on Windows:
1. Download wheel from: https://github.com/mrjbq7/ta-lib#windows
2. Install: `pip install TA_Lib-0.4.28-cp3x-cp3x-win_amd64.whl`

### 4. Test Development Version

```powershell
# Test CLI version
python main.py --help

# Test GUI version
python windows_app.py
```

---

## Building the Application

### Method 1: Automated Build (Recommended)

Simply run the build script:

```powershell
# Run the automated build script
build_windows.bat
```

This will:
1. ✅ Install build dependencies
2. ✅ Clean previous builds
3. ✅ Build executable with PyInstaller
4. ✅ Create installer (if Inno Setup is installed)

**Output:**
- Executable: `dist\TradingBotSimulator\TradingBotSimulator.exe`
- Installer: `Output\TradingBotSimulator-Setup.exe`

### Method 2: Manual Build

#### Step 1: Install PyInstaller

```powershell
pip install pyinstaller pywin32
```

#### Step 2: Build with PyInstaller

```powershell
# Clean previous builds
rmdir /s /q build dist

# Build using spec file
pyinstaller trading_bot.spec --clean --noconfirm
```

#### Step 3: Test the Executable

```powershell
# Navigate to dist folder
cd dist\TradingBotSimulator

# Run the executable
TradingBotSimulator.exe
```

---

## Testing the Application

### 1. Functional Testing

Test all major features:

#### Dashboard
- ✅ Application launches without errors
- ✅ Dashboard displays correctly
- ✅ Performance metrics show default values

#### Backtesting
1. Go to "Backtesting" tab
2. Enter symbol: `AAPL`
3. Select strategy: `MACD`
4. Set date range: Last 1 year
5. Click "Run Backtest"
6. ✅ Verify results display correctly

#### Paper Trading
1. Go to "Paper Trading" tab
2. Enter symbols: `AAPL,MSFT`
3. Select strategy: `RSI`
4. Click "Start Paper Trading"
5. ✅ Verify trading status updates
6. Click "Stop Trading"
7. ✅ Verify summary displays

#### Portfolio Optimization
1. Go to "Portfolio" tab
2. Enter symbols: `AAPL,MSFT,GOOGL,AMZN`
3. Select method: `Maximum Sharpe Ratio`
4. Click "Optimize Portfolio"
5. ✅ Verify optimal weights display

#### Machine Learning
1. Go to "Machine Learning" tab
2. Enter symbol: `AAPL`
3. Click "Train Model"
4. ✅ Verify training completes
5. Click "Predict"
6. ✅ Verify predictions display

#### Settings
1. Go to "Settings" tab
2. Modify trading parameters
3. Click "Save Settings"
4. ✅ Verify settings are saved

### 2. System Integration Testing

#### Windows 11 Features
- ✅ System tray icon appears
- ✅ Right-click tray icon shows menu
- ✅ Double-click tray icon shows window
- ✅ Windows notifications work
- ✅ Application appears in Windows notifications settings
- ✅ Taskbar icon displays correctly
- ✅ Alt+Tab shows application

#### File System
- ✅ Config file loads correctly
- ✅ Logs are created in AppData
- ✅ Data is cached properly

### 3. Performance Testing

- ✅ Application startup time < 5 seconds
- ✅ Memory usage < 500 MB during normal operation
- ✅ CPU usage reasonable during backtesting
- ✅ No memory leaks during extended use

### 4. Error Handling

- ✅ Invalid input handled gracefully
- ✅ Network errors display user-friendly messages
- ✅ Crash reports generated for unhandled exceptions
- ✅ Application recovers from errors

---

## Creating the Installer

### Prerequisites

Install Inno Setup:
1. Download from: https://jrsoftware.org/isinfo.php
2. Install with default options
3. Add to PATH (optional)

### Build Installer

#### Option 1: Via Build Script

The build script automatically creates the installer if Inno Setup is installed.

```powershell
build_windows.bat
```

#### Option 2: Manual Creation

```powershell
# Compile installer script
iscc installer.iss
```

**Output:** `Output\TradingBotSimulator-Setup.exe`

### Test Installer

1. **Run the installer**
   ```powershell
   Output\TradingBotSimulator-Setup.exe
   ```

2. **Verify installation steps:**
   - ✅ Welcome screen displays
   - ✅ License agreement shows
   - ✅ Installation directory selection works
   - ✅ Progress bar advances
   - ✅ Desktop shortcut created (if selected)
   - ✅ Start menu entry created
   - ✅ Application launches after installation

3. **Test installed application:**
   - ✅ Launch from Start menu
   - ✅ Launch from desktop shortcut
   - ✅ All features work
   - ✅ Settings persist between runs

4. **Test uninstallation:**
   - ✅ Uninstaller in Start menu works
   - ✅ Uninstaller in Control Panel works
   - ✅ Application files removed
   - ✅ Settings preserved (in AppData)

---

## Distribution

### 1. Digital Signature (Recommended)

For production distribution, sign your installer:

```powershell
# Sign with code signing certificate
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com TradingBotSimulator-Setup.exe
```

Benefits:
- ✅ Reduces Windows SmartScreen warnings
- ✅ Builds user trust
- ✅ Verifies authenticity

**Get a certificate from:**
- DigiCert
- Sectigo
- GlobalSign

### 2. Hosting Options

#### GitHub Releases
1. Create a new release on GitHub
2. Upload `TradingBotSimulator-Setup.exe`
3. Add release notes
4. Tag version (e.g., `v1.0.0`)

#### Website Download
- Host on your website
- Provide direct download link
- Include SHA-256 checksum

#### Microsoft Store (Advanced)
- Convert to MSIX package
- Submit to Microsoft Partner Center
- Requires developer account ($19/year)

### 3. Auto-Update Configuration

The application includes auto-update functionality. Configure:

1. Edit `src/utils/updater.py`
2. Set your GitHub repository URL:
   ```python
   update_url = "https://api.github.com/repos/YOUR_USERNAME/trading-bot-simulator/releases/latest"
   ```

3. Users will be notified when updates are available

---

## Troubleshooting

### Build Issues

#### Issue: PyInstaller fails with import errors

**Solution:**
```powershell
# Install missing packages
pip install missing-package-name

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

#### Issue: "Module not found" in built executable

**Solution:**
Edit `trading_bot.spec` and add to `hiddenimports`:
```python
hiddenimports = [
    'missing_module_name',
    ...
]
```

#### Issue: Large executable size

**Solution:**
- Normal size: 300-500 MB (includes Python, ML libraries)
- To reduce: Exclude unused packages in spec file
- Use UPX compression (already enabled)

### Runtime Issues

#### Issue: Application won't start

**Solutions:**
1. Check logs in: `%APPDATA%\Trading Bot Simulator\logs\`
2. Run from Command Prompt to see errors:
   ```powershell
   dist\TradingBotSimulator\TradingBotSimulator.exe
   ```
3. Install Visual C++ Redistributable

#### Issue: "Windows protected your PC" SmartScreen warning

**Solutions:**
1. Click "More info" → "Run anyway" (for testing)
2. For production: Sign the executable with code signing certificate

#### Issue: Antivirus false positive

**Solutions:**
1. Add exception in antivirus software
2. Sign the executable
3. Report false positive to antivirus vendor

### Performance Issues

#### Issue: Slow startup

**Causes:**
- First run: Data download
- Antivirus scanning
- ML model initialization

**Solutions:**
- Whitelist in antivirus
- Use SSD storage
- Close other applications

---

## Production Checklist

Before releasing to users:

### Code Quality
- ✅ All features tested and working
- ✅ No debug/print statements in production
- ✅ Error handling for all user inputs
- ✅ Logging configured properly
- ✅ Code reviewed and optimized

### Build Configuration
- ✅ Version number updated in:
  - `version_info.txt`
  - `installer.iss`
  - `windows_app.py`
- ✅ Application icon created and referenced
- ✅ All resources included
- ✅ Configuration file included

### Testing
- ✅ Clean install tested on fresh Windows 11
- ✅ All features work in built version
- ✅ Installer tested
- ✅ Uninstaller tested
- ✅ Settings persistence tested
- ✅ Performance acceptable

### Distribution
- ✅ Executable signed (recommended)
- ✅ Installer created and tested
- ✅ Release notes prepared
- ✅ Documentation updated
- ✅ Download location set up

### Legal & Documentation
- ✅ License file included
- ✅ Disclaimer about trading risks
- ✅ README updated
- ✅ User guide available
- ✅ Support contact provided

---

## Advanced Topics

### Creating MSIX Package for Microsoft Store

```powershell
# Install MSIX Packaging Tool from Microsoft Store
# Then use the GUI to convert your installer
```

### Continuous Integration

Example GitHub Actions workflow (`.github/workflows/build.yml`):

```yaml
name: Build Windows App

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pyinstaller
      - run: pyinstaller trading_bot.spec
      - uses: actions/upload-artifact@v2
        with:
          name: TradingBotSimulator
          path: dist/TradingBotSimulator
```

### Crash Reporting

The application includes crash reporting. Crashes are saved to:
```
%APPDATA%\Trading Bot Simulator\logs\crash_report_*.json
```

To implement remote crash reporting, integrate with services like:
- Sentry
- Raygun
- BugSnag

---

## Support

For issues or questions:

1. Check this guide
2. Review logs in AppData
3. Search GitHub issues
4. Create new issue with:
   - Windows version
   - Python version
   - Error message/logs
   - Steps to reproduce

---

## Version History

- **1.0.0** (2024) - Initial release
  - Full GUI application
  - Backtesting engine
  - Paper trading
  - Portfolio optimization
  - Machine learning predictions
  - Windows 11 integration

---

**Happy Building! 🚀**

For the latest updates and documentation, visit:
https://github.com/yourusername/trading-bot-simulator
