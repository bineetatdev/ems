#!/usr/bin/env bash
# BuilMirai MPC HVAC Dashboard — startup script
# Usage:  ./start.sh
# Env vars (all optional):
#   PORT            — listening port              (default: 8008)
#   ENERGYPLUS_DIR  — EnergyPlus install path     (default: /usr/local/EnergyPlus-25-2-0)
#   WORKERS         — uvicorn worker count         (default: 1, keep 1 for EnergyPlus thread safety)

set -euo pipefail

PORT="${PORT:-8008}"
ENERGYPLUS_DIR="${ENERGYPLUS_DIR:-/usr/local/EnergyPlus-25-2-0}"
WORKERS="${WORKERS:-1}"

export PORT
export ENERGYPLUS_DIR

echo "========================================"
echo "  BuilMirai MPC HVAC Dashboard"
echo "========================================"
echo "  Dashboard : http://0.0.0.0:${PORT}/"
echo "  API docs  : http://0.0.0.0:${PORT}/docs"
echo "  EnergyPlus: ${ENERGYPLUS_DIR}"
echo "========================================"

exec uvicorn api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  --no-access-log
