# PPE Detection System - PowerShell Fix
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PPE Detection System - PowerShell Fix" -ForegroundColor Green  
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location "C:\Users\lespo\Downloads\Safeconstruction"
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Test Python
Write-Host "Testing Python..." -ForegroundColor Blue
try {
    $pythonVersion = python --version
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting minimal web interface (bypasses OpenCV issues)..." -ForegroundColor Blue
Write-Host ""
Write-Host "✅ Web Interface: http://localhost:8080" -ForegroundColor Green
Write-Host "✅ Demo Page: http://localhost:8080/demo" -ForegroundColor Green  
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start the application
python minimal_app.py

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"