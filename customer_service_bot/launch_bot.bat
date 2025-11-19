@echo off
REM AI Customer Service Bot Launcher for Windows 11
REM This script launches the customer service bot GUI

echo ================================================================================
echo AI Customer Service Bot - Windows 11
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Starting Customer Service Bot...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the bot in GUI mode
python bot_controller.py --gui

REM Check if there was an error
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the bot
    echo Please check the error messages above
    pause
    exit /b 1
)

pause
