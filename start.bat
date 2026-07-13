@echo off
REM Von & Co — Skin Analyzer Launcher (Windows)

cd /d "%~dp0"

echo.
echo   Von & Co - AI Skin Analyzer
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo   Python 3 is required. Install from https://python.org
    pause
    exit /b 1
)

if not exist .env (
    if exist env.txt (
        copy env.txt .env >nul
        echo   Created .env from env.txt
        echo   Edit .env to add your ANTHROPIC_API_KEY for AI features
        echo.
    )
)

python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies...
    pip install -r requirements.txt
    echo.
)

set PORT=5002
for /f "tokens=2 delims==" %%a in ('findstr /b "PORT=" .env 2^>nul') do set PORT=%%a
if not defined PORT set PORT=5002
for /f "tokens=2 delims==" %%a in ('findstr /b "PORT=" env.txt 2^>nul') do set PORT=%%a

echo   Starting server on http://localhost:%PORT%
echo   Press Ctrl+C to stop
echo.

start /b cmd /c "timeout /t 2 >nul & start http://localhost:%PORT%"

python server.py
