# Full Stack Startup Script for Data Navigator Pro
# Starts both backend and frontend servers

Write-Host ""
Write-Host "========================================"
Write-Host "Data Navigator Pro - Full Stack Startup"
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Function to start server
function Start-Server {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Command,
        [int]$Port
    )
    
    Write-Host "Starting $Name (Port $Port)..." -ForegroundColor Yellow
    
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = "powershell"
    $pinfo.UseShellExecute = $true
    $pinfo.Arguments = "-NoExit -Command `"cd '$Path' ; $Command`""
    
    $p = [System.Diagnostics.Process]::Start($pinfo)
    Write-Host "✅ $Name started (PID: $($p.Id))" -ForegroundColor Green
}

# Start backend
Start-Server -Name "Backend Server" -Path "$scriptDir\backend" -Command "python run.py" -Port 8000

# Wait for backend to start
Write-Host ""
Write-Host "Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start frontend
Start-Server -Name "Frontend Server" -Path "$scriptDir\data-navigator-pro-main" -Command "npm run dev" -Port 5173

Write-Host ""
Write-Host "========================================"
Write-Host "✨ Servers Starting..." -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Both servers are starting in new windows..." -ForegroundColor Yellow
Write-Host ""
