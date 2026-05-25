# BuilMirai MPC HVAC Dashboard — PowerShell startup script
# Usage:  .\start.ps1
# Override defaults:
#   $env:PORT = "9000"; .\start.ps1

param(
    [int]$Port             = if ($env:PORT)           { [int]$env:PORT }           else { 8008 },
    [string]$EnergyPlusDir = if ($env:ENERGYPLUS_DIR) { $env:ENERGYPLUS_DIR }      else { "C:\EnergyPlusV25-2-0" },
    [string]$WeatherFile   = if ($env:WEATHER_FILE)   { $env:WEATHER_FILE }        else { "" }
)

$env:PORT = $Port
$env:ENERGYPLUS_DIR = $EnergyPlusDir
if ($WeatherFile) { $env:WEATHER_FILE = $WeatherFile }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BuilMirai MPC HVAC Dashboard" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Dashboard : http://0.0.0.0:$Port/" -ForegroundColor Green
Write-Host "  API docs  : http://0.0.0.0:$Port/docs" -ForegroundColor Green
Write-Host "  EnergyPlus: $EnergyPlusDir" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$uvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $uvicorn)) {
    Write-Host "ERROR: virtualenv not found. Run setup first:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\pip install fastapi uvicorn[standard] pydantic" -ForegroundColor Yellow
    exit 1
}

& $uvicorn api.main:app `
    --host 0.0.0.0 `
    --port $Port `
    --workers 1 `
    --no-access-log
