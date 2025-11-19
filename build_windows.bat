@echo off
REM Trading Bot Simulator - Windows Build Script
REM This script builds the Windows executable and installer

echo ================================================
echo Trading Bot Simulator - Windows Build Script
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version

echo.
echo [2/6] Installing/upgrading build dependencies...
pip install --upgrade pip setuptools wheel
pip install --upgrade pyinstaller pywin32

echo.
echo [3/6] Installing application dependencies...
pip install -r requirements.txt

echo.
echo [4/6] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [5/6] Building Windows executable with PyInstaller...
pyinstaller trading_bot.spec --clean --noconfirm

if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo [6/6] Build complete!
echo.
echo Executable location: dist\TradingBotSimulator\TradingBotSimulator.exe
echo.

REM Check if Inno Setup is available for installer creation
where iscc >nul 2>&1
if errorlevel 1 (
    echo.
    echo NOTE: Inno Setup not found. Skipping installer creation.
    echo To create an installer, install Inno Setup from:
    echo https://jrsoftware.org/isinfo.php
    echo Then run: iscc installer.iss
) else (
    echo.
    echo Creating Windows installer...
    iscc installer.iss
    if errorlevel 1 (
        echo WARNING: Installer creation failed
    ) else (
        echo Installer created: Output\TradingBotSimulator-Setup.exe
    )
)

echo.
echo ================================================
echo Build process completed successfully!
echo ================================================
pause
