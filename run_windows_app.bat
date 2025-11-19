@echo off
REM Trading Bot Simulator - Windows GUI Launcher
REM Quick launcher for the Windows application

echo ================================================
echo Trading Bot Simulator - Windows 11 Edition
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate
)

REM Check if dependencies are installed
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Launching Trading Bot Simulator...
echo.

REM Run the Windows application
python windows_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the error messages above
    pause
    exit /b 1
)
