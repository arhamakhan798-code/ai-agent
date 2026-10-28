@echo off
echo ============================================================
echo   TechMart AI Enterprise RAG Agent - Quick Start
echo ============================================================
echo.

cd /d "%~dp0"

REM Load environment variables from .env if present
if exist .env (
    for /f "usebackq tokens=*" %%a in (".env") do (
        set "%%a"
    )
)

echo [1] Starting FastAPI backend on http://localhost:5000
echo [2] Documentation available at http://localhost:5000/docs
echo [3] Open widget.html or visit http://localhost:5000 directly
echo.
echo Press Ctrl+C to stop.
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
pause
