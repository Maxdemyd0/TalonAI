@echo off
setlocal

cd /d "%~dp0"

if not exist ".\.venv\Scripts\python.exe" (
    echo Talon's virtual environment was not found at .\.venv\Scripts\python.exe
    echo Activate or create the venv first, then try again.
    pause
    exit /b 1
)

echo Training Talon on the GPU...
.\.venv\Scripts\python.exe -m talon.train --device cuda --block-size 256 --max-steps 1200 --output-dir artifacts/talon-base %*

if errorlevel 1 (
    echo.
    echo Talon training exited with an error.
    pause
    exit /b %errorlevel%
)

echo.
echo Talon training finished successfully.
pause

endlocal
