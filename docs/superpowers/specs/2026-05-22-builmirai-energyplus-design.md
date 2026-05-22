# BuilMirai MPC HVAC Dashboard — EnergyPlus Backend Design

**Date:** 2026-05-22
**Status:** Approved

## Overview

Replace the fake JavaScript computation in `builmirai_mpc_hvac_dashboard.html` with a real EnergyPlus 25.2 physics simulation engine. A FastAPI backend runs EnergyPlus via the `pyenergyplus` Python API in short-horizon co-simulation mode (4-hour prediction horizon, 15-min timesteps). A heuristic MPC layer computes optimal HVAC setpoints before each simulation run. The dashboard frontend calls `POST /optimize` and renders real simulation results.

## Architecture

Four layers:

```
Frontend (HTML dashboard)
  └─ HTTP REST (POST /optimize)
FastAPI (api/main.py)
  └─ Python call
Heuristic MPC (simulation/mpc.py)
  └─ Python call
EnergyPlus Engine (simulation/engine.py)  ← pyenergyplus Python API
```

### File Structure

```
demo-project/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + routes
│   └── models.py        # Pydantic request/response models
├── simulation/
│   ├── __init__.py
│   ├── engine.py        # EnergyPlus Python API wrapper
│   ├── mpc.py           # Heuristic MPC setpoint logic
│   ├── building.idf     # 5-zone office building model
│   └── weather.epw      # Standard EPW weather file
├── builmirai_mpc_hvac_dashboard.html  # Updated frontend
├── main.py              # Entry point: starts uvicorn
└── pyproject.toml       # Updated dependencies
```

## Building Model (IDF)

Single-story 5-zone office building, ~500 m².

### Zones

| Zone | Area | Occupancy Schedule | Temp Limits | Notes |
|------|------|--------------------|-------------|-------|
| Server Hall | 30 m² | 24/7 | 18–24°C | High IT internal gains (5 W/m²) |
| Open Plan | 200 m² | 08:00–18:00 variable | 22–26°C | Largest zone |
| Boardroom | 50 m² | 09:00–17:00 scheduled | 22–26°C | Dense occupancy during meetings |
| Reception | 40 m² | 08:00–19:00 low density | 22–26°C | Near-perimeter, solar gains |
| Lab A | 80 m² | 08:00–20:00 | 22–26°C | Moderate equipment gains |

### HVAC

- **AHU-1** — serves Server Hall + Lab A
- **AHU-2** — serves Open Plan + Boardroom + Reception
- **Chiller** — water-cooled with free-cooling economiser
- **Solar PV** — 30 kW peak, modelled as negative load offset on electricity meter

### EMS Actuators

- AHU-1 supply air temperature setpoint
- AHU-2 supply air temperature setpoint
- Chiller leaving water temperature setpoint
- Zone thermostat heating/cooling setpoints (all 5 zones)
- Occupancy schedule multiplier (driven by occupancy slider)
- Dry-bulb temperature weather override (driven by ext_temp slider)

### EMS Output Variables (collected each timestep)

- Zone air temperature — all 5 zones
- HVAC electricity demand (kW)
- Chiller COP
- Free-cooling fraction

## API Design

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/optimize` | Run MPC + EnergyPlus, return results |
| `GET` | `/health` | Check server + EnergyPlus availability |
| `GET` | `/scenarios/{name}` | Return preset inputs for a named scenario |

### POST /optimize

**Request:**
```json
{
  "occupancy": 70,
  "ext_temp": 24.0,
  "pv_kw": 14.0,
  "tariff": 11.0,
  "horizon_hours": 4
}
```

Constraints: `occupancy` 0–100, `ext_temp` 10–42, `pv_kw` 0–30, `tariff` 5–40, `horizon_hours` default 4.

`savings_pct` is computed by the MPC layer as: `round((baseline_power - mpc_power) / baseline_power * 100)`, where `baseline_power` is the simulated HVAC power with no setpoint optimisation (all setpoints at fixed defaults). The MPC layer estimates baseline power from the same EnergyPlus run with neutral setpoint inputs, using the first timestep before EMS injection takes effect.

**Response:**
```json
{
  "power_kw": 38.2,
  "savings_pct": 18,
  "avg_zone_temp": 23.4,
  "pv_contribution_pct": 37,
  "comfort_zones": 5,
  "zone_temps": {
    "Server Hall": 22.1,
    "Open Plan": 23.4,
    "Boardroom": 23.8,
    "Reception": 22.9,
    "Lab A": 23.6
  },
  "energy_forecast_kwh": [38, 35, 33, 31, 28, 26, 24],
  "setpoints": {
    "AHU-1 supply": "17.8°C",
    "AHU-2 supply": "18.2°C",
    "Chiller setpt": "6.8°C",
    "Free-cool %": "45%",
    "PV divert": "47%",
    "Demand limit": "46 kW"
  },
  "simulation_duration_s": 12.4
}
```

### GET /health

```json
{
  "status": "ok",
  "energyplus_version": "25.2",
  "idf_loaded": true
}
```

### GET /scenarios/{name}

Valid names: `normal`, `peak`, `heatwave`, `preheat`, `night`.

```json
{
  "name": "peak",
  "occupancy": 90,
  "ext_temp": 27.0,
  "pv_kw": 8.0,
  "tariff": 34.0
}
```

## EnergyPlus Engine (`simulation/engine.py`)

Each `/optimize` request creates a fresh EnergyPlus API state — stateless per request.

### Simulation Flow

1. Create new EnergyPlus state via `api.state_manager.new_state()`
2. Register callbacks:
   - `on_begin_timestep` — inject occupancy multiplier + ext_temp override via EMS actuators
   - `on_end_timestep` — collect zone temps + HVAC power, append to result list
   - `on_end_warmup` — mark warmup complete, begin data collection
3. Call `api.runtime.run_energyplus()` in thread pool executor (non-blocking)
4. Aggregate timestep data into `SimulationResult`

### Forecast Aggregation

The simulation collects 16 timestep values (4 hours × 15 min). These are subsampled to the 7 forecast points shown on the dashboard:

| Dashboard label | Timestep index |
|----------------|---------------|
| +15m | 1 |
| +30m | 2 |
| +45m | 3 |
| +1h | 4 |
| +2h | 8 |
| +3h | 12 |
| +4h | 16 |

### Thread Safety

EnergyPlus is not thread-safe across instances. Each request:
- Gets its own API state object
- Writes to a unique temp directory: `/tmp/ep_run_{uuid}`
- Is serialised via `asyncio.Lock` to prevent resource conflicts across concurrent requests

FastAPI offloads the blocking simulation to the default thread pool via `asyncio.get_event_loop().run_in_executor(None, ...)`.

## Heuristic MPC Layer (`simulation/mpc.py`)

Runs before the EnergyPlus simulation. Computes setpoints from 4 slider inputs using additive rules:

| Condition | Rule | Effect |
|-----------|------|--------|
| `tariff > 25` | AHU supply temp +2°C, reduce chiller load | Shift cooling from peak tariff |
| `pv_kw > 15` | Chiller setpoint −1°C, free-cool +20% | Exploit solar generation |
| `ext_temp > 35` | Maximise pre-cooling, tighten demand limit | Heatwave protection |
| `occupancy < 20` | Zone setpoints +1.5°C, reduce AHU flow | Night setback |
| `occupancy > 80` | Tighten zone setpoints, increase airflow | Dense occupancy comfort |

Base setpoints (before rules): AHU-1 supply 18°C, AHU-2 supply 18°C, chiller 7°C, free-cool 30%.

Output: `Setpoints` dataclass consumed by the engine's EMS callback.

## Frontend Changes

In `builmirai_mpc_hvac_dashboard.html`, the `runOptimise()` function replaces `computeState()` with a real API call:

```javascript
async function runOptimise() {
  // show loading state (existing UI)
  const res = await fetch('http://localhost:8000/optimize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      occupancy: state.occ,
      ext_temp: state.ext,
      pv_kw: state.pv,
      tariff: state.tariff
    })
  });
  const data = await res.json();
  // render with real data (existing render functions, adapted)
}
```

The `setScenario()` function will call `GET /scenarios/{name}` to fetch preset inputs, then trigger `runOptimise()`.

All existing rendering functions (`renderZones`, `renderForecast`, `renderSetpoints`) are adapted to accept the API response shape — the UI structure stays unchanged.

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.29",
  "pydantic>=2.0",
]
```

`pyenergyplus` is not pip-installable — it ships with EnergyPlus 25.2. The engine will locate it via the EnergyPlus installation path (typically `/usr/local/EnergyPlus-25-2-0/` on macOS).

## Out of Scope (for this iteration)

- WebSocket real-time streaming (future: replace REST with WS for 15s MPC cycle)
- True MPC (solver-based multi-trajectory optimisation)
- Authentication / multi-user support
- Production deployment / containerisation
- Historical data logging / time-series database
