@echo off
REM Run tests for AI Medical Diagnosis System

echo Running AI Medical Diagnosis System Tests...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run tests
echo Running pytest...
pytest -v --tb=short

echo.
echo Tests complete!
pause
