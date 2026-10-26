@echo off
echo ============================================================
echo   TechMart AI Agent - Quick Start
echo ============================================================
echo.

cd /d "%~dp0"

REM Load environment variables from .env if present
if exist .env (
    for /f "usebackq tokens=*" %%a in (".env") do (
        set "%%a"
    )
)

echo [1] Starting Flask backend on http://localhost:5000
echo [2] Open widget.html in your browser to use the chat
echo.
echo Press Ctrl+C to stop.
echo.

python app.py
pause
