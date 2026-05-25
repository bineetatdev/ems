# BuilMirai — Install as Windows Service using NSSM
# Run as Administrator in PowerShell
# Usage: .\install-windows-service.ps1

param(
    [string]$AppDir        = "C:\builmirai",
    [int]$Port             = 8008,
    [string]$EnergyPlusDir = "C:\EnergyPlusV25-2-0",
    [string]$ServiceName   = "BuilMirai"
)

# Check admin
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Run this script as Administrator." -ForegroundColor Red
    exit 1
}

# Check NSSM
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    Write-Host "NSSM not found. Installing via winget..." -ForegroundColor Yellow
    winget install --id NSSM.NSSM -e
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")
    $nssm = Get-Command nssm -ErrorAction SilentlyContinue
    if (-not $nssm) {
        Write-Host "ERROR: NSSM install failed. Download from https://nssm.cc/download" -ForegroundColor Red
        exit 1
    }
}

$uvicorn  = "$AppDir\.venv\Scripts\uvicorn.exe"
$logDir   = "$AppDir\logs"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

Write-Host "Installing service '$ServiceName'..." -ForegroundColor Cyan

nssm install  $ServiceName $uvicorn "api.main:app --host 0.0.0.0 --port $Port --workers 1 --no-access-log"
nssm set      $ServiceName AppDirectory   $AppDir
nssm set      $ServiceName AppEnvironmentExtra "PORT=$Port" "ENERGYPLUS_DIR=$EnergyPlusDir"
nssm set      $ServiceName AppStdout      "$logDir\builmirai.log"
nssm set      $ServiceName AppStderr      "$logDir\builmirai-error.log"
nssm set      $ServiceName AppRotateFiles 1
nssm set      $ServiceName AppRotateBytes 5242880
nssm set      $ServiceName Start          SERVICE_AUTO_START
nssm set      $ServiceName DisplayName    "BuilMirai MPC HVAC Dashboard"
nssm set      $ServiceName Description   "AI-MPC Building Energy Management — FastAPI + EnergyPlus 25.2"

nssm start $ServiceName

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Service '$ServiceName' installed and started!" -ForegroundColor Green
Write-Host "  Dashboard: http://localhost:$Port/" -ForegroundColor Green
Write-Host ""
Write-Host "  Manage with:" -ForegroundColor Yellow
Write-Host "    nssm start   $ServiceName" -ForegroundColor Yellow
Write-Host "    nssm stop    $ServiceName" -ForegroundColor Yellow
Write-Host "    nssm restart $ServiceName" -ForegroundColor Yellow
Write-Host "    nssm remove  $ServiceName confirm" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
