@echo off
:: BuilMirai MPC HVAC Dashboard — Windows startup script
:: Usage:  start.bat
:: Override defaults by setting env vars before running:
::   set PORT=9000
::   set ENERGYPLUS_DIR=C:\EnergyPlusV25-2-0
::   start.bat

if not defined PORT set PORT=8008
if not defined ENERGYPLUS_DIR set ENERGYPLUS_DIR=C:\EnergyPlusV25-2-0

echo ========================================
echo   BuilMirai MPC HVAC Dashboard
echo ========================================
echo   Dashboard : http://0.0.0.0:%PORT%/
echo   API docs  : http://0.0.0.0:%PORT%/docs
echo   EnergyPlus: %ENERGYPLUS_DIR%
echo ========================================

.venv\Scripts\uvicorn.exe api.main:app ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --workers 1 ^
  --no-access-log
