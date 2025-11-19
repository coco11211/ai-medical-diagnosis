# AI Medical Diagnosis System - Windows Service Installation
# Run this script as Administrator to install as Windows Service

param(
    [string]$ServiceName = "AIMedicalDiagnosis",
    [string]$DisplayName = "AI Medical Diagnosis System",
    [string]$Description = "AI-powered medical diagnosis system with REST API"
)

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Please right-click and select 'Run as Administrator'" -ForegroundColor Yellow
    Pause
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Medical Diagnosis System" -ForegroundColor Cyan
Write-Host "Windows Service Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get current directory
$currentDir = Get-Location
$pythonPath = Join-Path $currentDir "venv\Scripts\python.exe"
$scriptPath = Join-Path $currentDir "src\api\main.py"

# Verify Python exists
if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Python virtual environment not found at: $pythonPath" -ForegroundColor Red
    Write-Host "Please run setup_windows.bat first" -ForegroundColor Yellow
    Pause
    exit 1
}

# Verify script exists
if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: Main script not found at: $scriptPath" -ForegroundColor Red
    Pause
    exit 1
}

# Check if service already exists
$existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existingService) {
    Write-Host "Service already exists. Removing existing service..." -ForegroundColor Yellow
    Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    sc.exe delete $ServiceName
    Start-Sleep -Seconds 2
}

# Install NSSM (Non-Sucking Service Manager)
Write-Host "Installing service using NSSM..." -ForegroundColor Green

# Download NSSM if not present
$nssmPath = Join-Path $currentDir "scripts\nssm.exe"
if (-not (Test-Path $nssmPath)) {
    Write-Host "NSSM not found. Please download NSSM from https://nssm.cc/download" -ForegroundColor Yellow
    Write-Host "Extract nssm.exe to the scripts folder and run this script again" -ForegroundColor Yellow
    Pause
    exit 1
}

# Create service using NSSM
& $nssmPath install $ServiceName $pythonPath "-m" "src.api.main"
& $nssmPath set $ServiceName AppDirectory $currentDir
& $nssmPath set $ServiceName DisplayName $DisplayName
& $nssmPath set $ServiceName Description $Description
& $nssmPath set $ServiceName Start SERVICE_AUTO_START
& $nssmPath set $ServiceName AppStdout (Join-Path $currentDir "logs\service_stdout.log")
& $nssmPath set $ServiceName AppStderr (Join-Path $currentDir "logs\service_stderr.log")

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Service installed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service Name: $ServiceName" -ForegroundColor White
Write-Host "Display Name: $DisplayName" -ForegroundColor White
Write-Host ""
Write-Host "To start the service:" -ForegroundColor Yellow
Write-Host "  Start-Service -Name $ServiceName" -ForegroundColor White
Write-Host ""
Write-Host "To stop the service:" -ForegroundColor Yellow
Write-Host "  Stop-Service -Name $ServiceName" -ForegroundColor White
Write-Host ""
Write-Host "To remove the service:" -ForegroundColor Yellow
Write-Host "  .\scripts\uninstall_service.ps1" -ForegroundColor White
Write-Host ""

Pause
