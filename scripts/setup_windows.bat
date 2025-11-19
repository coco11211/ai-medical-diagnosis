@echo off
REM AI Medical Diagnosis System - Windows 11 Setup Script
REM This script sets up the environment for Windows 11

echo ========================================
echo AI Medical Diagnosis System
echo Windows 11 Setup Script
echo ========================================
echo.

REM Check Python installation
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python found!
python --version
echo.

REM Check Python version
echo [2/8] Verifying Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.9 or higher is required
    pause
    exit /b 1
)
echo Python version OK!
echo.

REM Create virtual environment
echo [3/8] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [6/8] Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

REM Download NLTK data
echo [7/8] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
if errorlevel 1 (
    echo WARNING: Failed to download NLTK data
    echo You may need to download manually later
) else (
    echo NLTK data downloaded!
)
echo.

REM Create necessary directories
echo [8/8] Creating directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\models mkdir data\models
if not exist logs mkdir logs
if not exist output mkdir output
echo Directories created!
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created! Please edit it with your configuration.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your configuration
echo 2. Edit config/config.yaml if needed
echo 3. Run: python -m src.api.main
echo.
echo To start using the system:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Start API server: python -m src.api.main
echo 3. Open browser: http://localhost:8000/docs
echo.
pause
