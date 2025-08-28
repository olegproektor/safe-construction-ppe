@echo off
echo ============================================================
echo PPE Detection System - Starting Server
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking Python environment...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! Please install Python or activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import fastapi, uvicorn; print('FastAPI and Uvicorn available')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: FastAPI or Uvicorn not installed!
    echo Installing dependencies...
    pip install fastapi uvicorn
)

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8080
echo Demo cameras page: http://localhost:8080/demo
echo API documentation: http://localhost:8080/docs
echo.
echo Press Ctrl+C to stop the server
echo ============================================================

uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

echo.
echo Server stopped. Press any key to close...
pause > nul