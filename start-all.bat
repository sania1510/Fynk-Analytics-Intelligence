@echo off
REM Full Stack Startup Script for Data Navigator Pro
REM Starts both backend and frontend servers

echo.
echo ========================================
echo Data Navigator Pro - Full Stack Startup
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Start backend in a new terminal
echo Starting Backend Server (Port 8000)...
start cmd /k cd /d "%SCRIPT_DIR%backend" ^& python run.py

REM Wait a few seconds for backend to start
timeout /t 3 /nobreak

REM Start frontend in a new terminal
echo Starting Frontend Server (Port 5173)...
start cmd /k cd /d "%SCRIPT_DIR%data-navigator-pro-main" ^& npm run dev

echo.
echo ========================================
echo ✅ Servers Starting...
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in either terminal to stop
echo.
pause
