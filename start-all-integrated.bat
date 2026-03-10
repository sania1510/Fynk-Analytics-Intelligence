@echo off
REM Data Navigator Pro - Quick Start Script for Windows

echo.
echo ==================================================
echo  Data Navigator Pro - Full Project Integration
echo ==================================================
echo.
echo Starting both frontend and backend servers...
echo.

REM Store current directory
set ROOT_DIR=%cd%

REM Check if backend folder exists
if not exist "backend" (
    echo ERROR: backend folder not found in current directory
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if frontend folder exists
if not exist "data-navigator-pro-main" (
    echo ERROR: data-navigator-pro-main folder not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo.
echo [1/2] Starting Backend (FastAPI on port 8000)...
echo ====================================================
start cmd /k "cd backend && python run.py"

echo.
echo [2/2] Starting Frontend (Vite on port 5173)...
echo ====================================================
start cmd /k "cd data-navigator-pro-main && npm run dev"

echo.
echo ==================================================
echo SUCCESS! Both servers are starting...
echo ==================================================
echo.
echo Frontend:  http://localhost:5173
echo Backend:   http://localhost:8000
echo API Docs:  http://localhost:8000/api/docs
echo.
echo Close the command windows to stop the servers.
echo.
pause
