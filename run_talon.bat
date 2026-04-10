@echo off
setlocal

cd /d "%~dp0"

if not exist ".\.venv\Scripts\python.exe" (
    echo Talon's virtual environment was not found at .\.venv\Scripts\python.exe
    echo Activate or create the venv first, then try again.
    pause
    exit /b 1
)

echo Starting Talon chat...
.\.venv\Scripts\python.exe -m talon.gui --checkpoint-dir artifacts/talon-base --web --show-sources --extractive-only %*

if errorlevel 1 (
    echo.
    echo Talon exited with an error.
    pause
    exit /b %errorlevel%
)

endlocal
