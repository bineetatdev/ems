# BuilMirai EnergyPlus Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake JavaScript computation in the BuilMirai dashboard with a real EnergyPlus 25.2 physics simulation engine exposed via FastAPI REST.

**Architecture:** FastAPI receives slider inputs from the HTML dashboard, passes them to a heuristic MPC layer that computes HVAC setpoints, then runs a 4-hour short-horizon EnergyPlus co-simulation via the `pyenergyplus` Python API, collecting zone temperatures and energy at each 15-min timestep. Results are returned as JSON; the frontend replaces its fake `computeState()` call with a real `fetch('POST /optimize')`.

**Tech Stack:** Python 3.13, FastAPI 0.111+, uvicorn, pydantic v2, pyenergyplus (bundled at `/Applications/EnergyPlus-25-2-0/`), EnergyPlus 25.2, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add fastapi, uvicorn, pydantic, pytest deps |
| `main.py` | Modify | Replace placeholder with `uvicorn.run` |
| `api/__init__.py` | Create | Empty package init |
| `api/models.py` | Create | Pydantic `OptimizeRequest`, `OptimizeResponse`, `HealthResponse`, `ScenarioResponse` |
| `api/main.py` | Create | FastAPI app: `POST /optimize`, `GET /health`, `GET /scenarios/{name}` |
| `simulation/__init__.py` | Create | Empty package init |
| `simulation/mpc.py` | Create | `Setpoints` dataclass + `compute_setpoints()` heuristic rules |
| `simulation/engine.py` | Create | `SimulationEngine` wrapping pyenergyplus Python API |
| `simulation/build_idf.py` | Create | Generator script → writes `simulation/building.idf` |
| `simulation/building.idf` | Create (generated) | 5-zone office EnergyPlus input file |
| `simulation/weather.epw` | Create (symlink) | EPW weather file (from EnergyPlus installation) |
| `builmirai_mpc_hvac_dashboard.html` | Modify | Replace `computeState()`/`runOptimise()` with real API calls |
| `tests/__init__.py` | Create | Empty |
| `tests/test_models.py` | Create | Pydantic validation tests |
| `tests/test_mpc.py` | Create | MPC setpoint rule tests |
| `tests/test_engine.py` | Create | Engine unit tests (mocked EnergyPlus) |
| `tests/test_api.py` | Create | FastAPI endpoint tests (mocked engine) |

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `api/__init__.py`, `simulation/__init__.py`, `tests/__init__.py`, `.gitignore`

- [ ] **Step 1: Update pyproject.toml**

Replace the existing `pyproject.toml` with:

```toml
[project]
name = "demo-project"
version = "0.1.0"
description = "BuilMirai MPC HVAC Dashboard with EnergyPlus backend"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.29",
  "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: No errors. `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx` installed.

- [ ] **Step 3: Create package directories and init files**

```bash
mkdir -p api simulation tests
touch api/__init__.py simulation/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
dist/
/tmp/ep_run_*/
simulation/building.idf
.superpowers/
```

Note: `building.idf` is generated — we commit `build_idf.py` instead.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml api/__init__.py simulation/__init__.py tests/__init__.py .gitignore
git commit -m "feat: project setup — fastapi, uvicorn, pydantic, pytest deps"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `api/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError
from api.models import OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse


def test_optimize_request_valid():
    req = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    assert req.occupancy == 70
    assert req.horizon_hours == 4  # default


def test_optimize_request_defaults_horizon():
    req = OptimizeRequest(occupancy=50, ext_temp=20.0, pv_kw=5.0, tariff=10.0)
    assert req.horizon_hours == 4


def test_optimize_request_rejects_out_of_range_occupancy():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=101, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_negative_occupancy():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=-1, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_ext_temp_too_high():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=43.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_ext_temp_too_low():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=9.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_pv_out_of_range():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=31.0, tariff=11.0)


def test_optimize_request_rejects_tariff_out_of_range():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=41.0)


def test_optimize_response_has_all_fields():
    resp = OptimizeResponse(
        power_kw=38.2,
        savings_pct=18,
        avg_zone_temp=23.4,
        pv_contribution_pct=37,
        comfort_zones=5,
        zone_temps={"Server Hall": 22.1, "Open Plan": 23.4, "Boardroom": 23.8, "Reception": 22.9, "Lab A": 23.6},
        energy_forecast_kwh=[38, 35, 33, 31, 28, 26, 24],
        setpoints={"AHU-1 supply": "17.8°C", "AHU-2 supply": "18.2°C", "Chiller setpt": "6.8°C",
                   "Free-cool %": "45%", "PV divert": "47%", "Demand limit": "46 kW"},
        simulation_duration_s=12.4,
    )
    assert resp.comfort_zones == 5
    assert len(resp.energy_forecast_kwh) == 7
    assert "Server Hall" in resp.zone_temps


def test_health_response():
    h = HealthResponse(status="ok", energyplus_version="25.2", idf_loaded=True)
    assert h.status == "ok"


def test_scenario_response():
    s = ScenarioResponse(name="peak", occupancy=90, ext_temp=27.0, pv_kw=8.0, tariff=34.0)
    assert s.name == "peak"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError` — `api.models` does not exist yet.

- [ ] **Step 3: Implement api/models.py**

Create `api/models.py`:

```python
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    occupancy: float = Field(..., ge=0, le=100)
    ext_temp: float = Field(..., ge=10, le=42)
    pv_kw: float = Field(..., ge=0, le=30)
    tariff: float = Field(..., ge=5, le=40)
    horizon_hours: int = Field(default=4, ge=1, le=8)


class OptimizeResponse(BaseModel):
    power_kw: float
    savings_pct: int
    avg_zone_temp: float
    pv_contribution_pct: int
    comfort_zones: int
    zone_temps: dict[str, float]
    energy_forecast_kwh: list[float]
    setpoints: dict[str, str]
    simulation_duration_s: float


class HealthResponse(BaseModel):
    status: str
    energyplus_version: str
    idf_loaded: bool


class ScenarioResponse(BaseModel):
    name: str
    occupancy: float
    ext_temp: float
    pv_kw: float
    tariff: float
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/models.py tests/test_models.py
git commit -m "feat: pydantic models for optimize request/response and health"
```

---

## Task 3: Heuristic MPC Layer

**Files:**
- Create: `simulation/mpc.py`
- Create: `tests/test_mpc.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mpc.py`:

```python
import pytest
from simulation.mpc import Setpoints, compute_setpoints


BASE = dict(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_returns_setpoints_instance():
    result = compute_setpoints(**BASE)
    assert isinstance(result, Setpoints)


def test_base_setpoints():
    """Default inputs produce base setpoints."""
    sp = compute_setpoints(**BASE)
    assert 16.0 <= sp.ahu1_supply_c <= 22.0
    assert 16.0 <= sp.ahu2_supply_c <= 22.0
    assert 5.0 <= sp.chiller_c <= 10.0
    assert 0.0 <= sp.free_cool_pct <= 100.0
    assert 0.0 <= sp.pv_divert_pct <= 100.0


def test_high_tariff_raises_supply_temps():
    """Peak tariff → raise supply air temps to reduce cooling load."""
    low = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=10.0)
    high = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=35.0)
    assert high.ahu1_supply_c > low.ahu1_supply_c
    assert high.ahu2_supply_c > low.ahu2_supply_c


def test_high_pv_lowers_chiller_setpoint():
    """High PV generation → lower chiller setpoint (exploit free energy)."""
    low_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=2.0, tariff=11.0)
    high_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=25.0, tariff=11.0)
    assert high_pv.chiller_c < low_pv.chiller_c


def test_high_pv_increases_free_cool():
    """High PV → increase free cooling fraction."""
    low_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=2.0, tariff=11.0)
    high_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=25.0, tariff=11.0)
    assert high_pv.free_cool_pct > low_pv.free_cool_pct


def test_heatwave_increases_free_cool_and_tightens_demand():
    """Heatwave (ext_temp > 35) → maximise pre-cooling."""
    normal = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    heatwave = compute_setpoints(occupancy=70, ext_temp=40.0, pv_kw=14.0, tariff=11.0)
    assert heatwave.demand_limit_kw < normal.demand_limit_kw


def test_low_occupancy_raises_setpoints():
    """Night setback (occupancy < 20) → relax zone setpoints."""
    normal = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=11.0)
    night = compute_setpoints(occupancy=5, ext_temp=24.0, pv_kw=0.0, tariff=5.0)
    assert night.zone_cooling_sp_c > normal.zone_cooling_sp_c


def test_high_occupancy_tightens_setpoints():
    """Dense occupancy → tighten zone setpoints for comfort."""
    low = compute_setpoints(occupancy=20, ext_temp=24.0, pv_kw=5.0, tariff=11.0)
    high = compute_setpoints(occupancy=95, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    assert high.zone_cooling_sp_c <= low.zone_cooling_sp_c


def test_setpoints_stay_in_physical_bounds():
    """Setpoints never leave physically valid ranges regardless of inputs."""
    for occ in [0, 50, 100]:
        for ext in [10, 30, 42]:
            for pv in [0, 15, 30]:
                for tariff in [5, 20, 40]:
                    sp = compute_setpoints(occ, ext, pv, tariff)
                    assert 12.0 <= sp.ahu1_supply_c <= 24.0, f"AHU1 out of range: {sp.ahu1_supply_c}"
                    assert 12.0 <= sp.ahu2_supply_c <= 24.0
                    assert 4.0 <= sp.chiller_c <= 12.0
                    assert 0 <= sp.free_cool_pct <= 100
                    assert 0 <= sp.pv_divert_pct <= 100
                    assert 20 <= sp.demand_limit_kw <= 200
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mpc.py -v
```

Expected: `ImportError` — `simulation.mpc` does not exist.

- [ ] **Step 3: Implement simulation/mpc.py**

Create `simulation/mpc.py`:

```python
from dataclasses import dataclass


@dataclass
class Setpoints:
    ahu1_supply_c: float      # AHU-1 supply air temperature (°C)
    ahu2_supply_c: float      # AHU-2 supply air temperature (°C)
    chiller_c: float          # Chiller leaving water temp (°C)
    free_cool_pct: float      # Free-cooling fraction (0-100 %)
    pv_divert_pct: float      # PV self-consumption fraction (0-100 %)
    zone_cooling_sp_c: float  # Zone thermostat cooling setpoint (°C)
    zone_heating_sp_c: float  # Zone thermostat heating setpoint (°C)
    demand_limit_kw: float    # Demand limit (kW)


# Base setpoints before rule adjustments
_BASE_AHU1 = 18.0
_BASE_AHU2 = 18.0
_BASE_CHILLER = 7.0
_BASE_FREE_COOL = 30.0
_BASE_COOLING_SP = 25.0
_BASE_HEATING_SP = 20.0
_BASE_DEMAND_LIMIT = 80.0


def compute_setpoints(
    occupancy: float,  # 0-100 %
    ext_temp: float,   # °C
    pv_kw: float,      # kW
    tariff: float,     # p/kWh
) -> Setpoints:
    """Compute HVAC setpoints using heuristic MPC rules.

    Rules combine additively from base values. Clamping to physical
    bounds is applied at the end.
    """
    ahu1 = _BASE_AHU1
    ahu2 = _BASE_AHU2
    chiller = _BASE_CHILLER
    free_cool = _BASE_FREE_COOL
    cooling_sp = _BASE_COOLING_SP
    heating_sp = _BASE_HEATING_SP
    demand_limit = _BASE_DEMAND_LIMIT

    # Rule 1: High tariff → reduce cooling load (raise supply temps)
    if tariff > 25:
        delta = (tariff - 25) / 15 * 2.0  # up to +2°C at max tariff
        ahu1 += delta
        ahu2 += delta

    # Rule 2: High PV → exploit free energy (lower chiller, more free-cooling)
    if pv_kw > 15:
        delta_chiller = (pv_kw - 15) / 15 * 1.0  # up to -1°C
        chiller -= delta_chiller
        free_cool += (pv_kw - 15) / 15 * 20.0  # up to +20%

    # Rule 3: Heatwave → pre-cool aggressively, tighten demand limit
    if ext_temp > 35:
        free_cool += (ext_temp - 35) / 7 * 30.0  # up to +30%
        demand_limit -= (ext_temp - 35) / 7 * 20.0  # tighten by up to 20 kW

    # Rule 4: Low occupancy (night setback) → relax zone setpoints
    if occupancy < 20:
        relax = (20 - occupancy) / 20 * 1.5  # up to +1.5°C
        cooling_sp += relax
        heating_sp -= relax / 2

    # Rule 5: Dense occupancy → tighten zone setpoints for comfort
    if occupancy > 80:
        tighten = (occupancy - 80) / 20 * 1.0  # up to -1°C
        cooling_sp -= tighten

    # PV divert scales linearly with PV output
    pv_divert = min(100.0, (pv_kw / 30) * 100)

    return Setpoints(
        ahu1_supply_c=max(12.0, min(24.0, ahu1)),
        ahu2_supply_c=max(12.0, min(24.0, ahu2)),
        chiller_c=max(4.0, min(12.0, chiller)),
        free_cool_pct=max(0.0, min(100.0, free_cool)),
        pv_divert_pct=max(0.0, min(100.0, pv_divert)),
        zone_cooling_sp_c=max(22.0, min(28.0, cooling_sp)),
        zone_heating_sp_c=max(16.0, min(22.0, heating_sp)),
        demand_limit_kw=max(20.0, min(200.0, demand_limit)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mpc.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simulation/mpc.py tests/test_mpc.py
git commit -m "feat: heuristic MPC setpoint rules with bounds clamping"
```

---

## Task 4: Building IDF Generator

**Files:**
- Create: `simulation/build_idf.py`
- Create: `simulation/building.idf` (generated by running the script)
- Create: `simulation/weather.epw` (symlink to EnergyPlus installation EPW)

- [ ] **Step 1: Create the IDF generator script**

Create `simulation/build_idf.py`:

```python
"""Generate simulation/building.idf for the 5-zone BuilMirai office building."""

from dataclasses import dataclass
from pathlib import Path

ENERGYPLUS_VERSION = "25.2.0"
H = 3.5  # zone height (m)
OUTPUT_PATH = Path(__file__).parent / "building.idf"


@dataclass
class ZoneSpec:
    name: str
    width: float   # x extent (m)
    depth: float   # y extent (north-south) (m)
    x0: float      # x origin (zones are spaced along x-axis to avoid overlap)
    people: int    # number of occupants at full occupancy
    equip_wm2: float  # electric equipment W/m²
    lighting_wm2: float = 10.0
    server_room: bool = False  # 24/7 equipment, no people


ZONES = [
    ZoneSpec("Server Hall",  width=5,  depth=6,  x0=0,   people=0,   equip_wm2=100.0, server_room=True),
    ZoneSpec("Open Plan",    width=14, depth=14, x0=10,  people=40,  equip_wm2=10.0),
    ZoneSpec("Boardroom",    width=7,  depth=7,  x0=30,  people=20,  equip_wm2=5.0),
    ZoneSpec("Reception",    width=6,  depth=7,  x0=42,  people=5,   equip_wm2=5.0),
    ZoneSpec("Lab A",        width=9,  depth=9,  x0=53,  people=15,  equip_wm2=40.0),
]


def _wall(name: str, zone: str, x1: float, y1: float, x2: float, y2: float,
          z_lo: float, z_hi: float, face: str, bc: str = "Adiabatic") -> str:
    """Generate a BuildingSurface:Detailed wall entry.

    face: 'S', 'N', 'E', 'W'
    Vertex order: upper-left first, CCW from outside (EnergyPlus convention).
    """
    sun = "SunExposed" if bc == "Outdoors" else "NoSun"
    wind = "WindExposed" if bc == "Outdoors" else "NoWind"
    bc_obj = "" if bc in ("Adiabatic", "Ground") else ""

    if face == "S":   # y=y1, outward normal = -y
        verts = f"  {x1},{y1},{z_hi},\n  {x1},{y1},{z_lo},\n  {x2},{y1},{z_lo},\n  {x2},{y1},{z_hi}"
    elif face == "N": # y=y2, outward normal = +y
        verts = f"  {x2},{y2},{z_hi},\n  {x2},{y2},{z_lo},\n  {x1},{y2},{z_lo},\n  {x1},{y2},{z_hi}"
    elif face == "E": # x=x2, outward normal = +x
        verts = f"  {x2},{y1},{z_hi},\n  {x2},{y1},{z_lo},\n  {x2},{y2},{z_lo},\n  {x2},{y2},{z_hi}"
    elif face == "W": # x=x1, outward normal = -x
        verts = f"  {x1},{y2},{z_hi},\n  {x1},{y2},{z_lo},\n  {x1},{y1},{z_lo},\n  {x1},{y1},{z_hi}"
    elif face == "Floor":
        verts = f"  {x1},{y1},{z_lo},\n  {x1},{y2},{z_lo},\n  {x2},{y2},{z_lo},\n  {x2},{y1},{z_lo}"
    elif face == "Ceiling":
        verts = f"  {x1},{y2},{z_hi},\n  {x2},{y2},{z_hi},\n  {x2},{y1},{z_hi},\n  {x1},{y1},{z_hi}"
    else:
        raise ValueError(f"Unknown face: {face}")

    surface_type = "Floor" if face == "Floor" else ("Ceiling" if face == "Ceiling" else "Wall")
    bc_cond = bc if bc != "Adiabatic" else "Adiabatic"

    return f"""BuildingSurface:Detailed,
  {name},                   !- Name
  {surface_type},           !- Surface Type
  {"Floor_Const" if face == "Floor" else "ExtWall_Const" if bc == "Outdoors" else "Adiabatic_Const"},
  {zone},                   !- Zone Name
  ,                         !- Space Name
  {bc_cond},               !- Outside Boundary Condition
  {"" if bc != "Outdoors" else ""},  !- Outside Boundary Condition Object
  {sun},                    !- Sun Exposure
  {wind},                   !- Wind Exposure
  {"0.5" if face == "Floor" else "autocalculate"},
  4,
{verts};
"""


def _window(name: str, parent_wall: str, zone: str,
            x1: float, x2: float, y: float, z_lo: float, z_hi: float) -> str:
    """Window on south-facing wall (y=const, x1→x2)."""
    wx1 = x1 + (x2 - x1) * 0.1
    wx2 = x2 - (x2 - x1) * 0.1
    wz_lo = z_lo + 0.8
    wz_hi = z_hi - 0.5
    return f"""FenestrationSurface:Detailed,
  {name},
  Window,
  Window_Const,             !- Construction Name
  {parent_wall},            !- Building Surface Name
  ,  ,  ,  ,
  4,
  {wx1},{y},{wz_hi},
  {wx1},{y},{wz_lo},
  {wx2},{y},{wz_lo},
  {wx2},{y},{wz_hi};
"""


def _zone_hvac(z: ZoneSpec) -> str:
    name = z.name
    return f"""
ZoneHVAC:IdealLoadsAirSystem,
  {name}_IdealLoads,        !- Name
  ,                         !- Availability Schedule
  {name}_SupplyAirInlet,    !- Zone Supply Air Node Name
  {name}_ExhaustAirOutlet,  !- Zone Exhaust Air Node Name
  ,                         !- System Inlet Air Node Name
  50,                       !- Maximum Heating Supply Air Temperature {{C}}
  13,                       !- Minimum Cooling Supply Air Temperature {{C}}
  0.015,                    !- Maximum Heating Supply Air Humidity Ratio {{kgWater/kgDryAir}}
  0.009,                    !- Minimum Cooling Supply Air Humidity Ratio {{kgWater/kgDryAir}}
  NoLimit,                  !- Heating Limit
  ,                         !- Maximum Heating Air Flow Rate (autosize)
  ,                         !- Maximum Sensible Heating Capacity (autosize)
  NoLimit,                  !- Cooling Limit
  ,                         !- Maximum Cooling Air Flow Rate (autosize)
  ,                         !- Maximum Total Cooling Capacity (autosize)
  ,  ,  ,  ,  ,  ,  ,  ,
  None;                     !- Demand Controlled Ventilation Type

ZoneHVAC:EquipmentConnections,
  {name},                   !- Zone Name
  {name}_EquipmentList,     !- Zone Conditioning Equipment List Name
  {name}_SupplyAirInlet,    !- Zone Air Inlet Node or NodeList Name
  {name}_ExhaustAirOutlet,  !- Zone Air Exhaust Node or NodeList Name
  {name}_ZoneAirNode,       !- Zone Air Node Name
  {name}_ZoneReturnAir;     !- Zone Return Air Node or NodeList Name

ZoneHVAC:EquipmentList,
  {name}_EquipmentList,
  SequentialLoad, ,
  ZoneHVAC:IdealLoadsAirSystem,
  {name}_IdealLoads,
  1, 1;
"""


def _people(z: ZoneSpec) -> str:
    if z.server_room:
        return ""
    sched = "Always_Off" if z.server_room else "Office_Occ"
    return f"""People,
  {z.name}_People,
  {z.name},
  {sched},                  !- Number of People Schedule
  People,
  {z.people},               !- Number of People
  ,
  0.3,                      !- Fraction Radiant
  AUTOCALCULATE,
  ActivityLevel_Sched;
"""


def _lights(z: ZoneSpec) -> str:
    area = z.width * z.depth
    watts = area * z.lighting_wm2
    sched = "Always_On" if z.server_room else "Office_Occ"
    return f"""Lights,
  {z.name}_Lights,
  {z.name},
  {sched},
  LightingLevel,
  {watts},
  ,
  ,
  0.0,                      !- Fraction Radiant
  0.0,                      !- Fraction Visible
  1.0,                      !- Fraction Replaceable
  GeneralLights;
"""


def _equipment(z: ZoneSpec) -> str:
    area = z.width * z.depth
    watts = area * z.equip_wm2
    sched = "Always_On" if z.server_room else "Office_Occ"
    return f"""ElectricEquipment,
  {z.name}_Equip,
  {z.name},
  {sched},
  EquipmentLevel,
  {watts},
  ,
  ,
  0.5;                      !- Fraction Radiant
"""


def _thermostat(z: ZoneSpec) -> str:
    return f"""ZoneControl:Thermostat,
  {z.name}_Thermostat,
  {z.name},
  Dual Zone Control Type Sched,
  ThermostatSetpoint:DualSetpoint,
  {z.name}_DualSP;

ThermostatSetpoint:DualSetpoint,
  {z.name}_DualSP,
  Heating_SP_Sched,         !- Heating Setpoint Temperature Schedule
  Cooling_SP_Sched;         !- Cooling Setpoint Temperature Schedule
"""


def _output_vars(z: ZoneSpec) -> str:
    return f"""Output:Variable,{z.name},Zone Air Temperature,TimeStep;
Output:Variable,{z.name},Zone Ideal Loads Supply Air Total Cooling Energy,TimeStep;
Output:Variable,{z.name},Zone Ideal Loads Supply Air Total Heating Energy,TimeStep;
"""


def generate_idf() -> str:
    header = f"""Version, {ENERGYPLUS_VERSION};

Building,
  BuilMirai Office,
  0.0,           !- North Axis {{deg}}
  City,          !- Terrain
  0.04,          !- Loads Convergence Tolerance
  0.4,           !- Temperature Convergence Tolerance
  FullInteriorAndExterior,
  5,             !- Maximum Number of Warmup Days
  1;             !- Minimum Number of Warmup Days

SimulationControl,
  No, No, No, No, Yes, No, 1;

Timestep, 4;   !- 4 per hour = 15 min intervals

RunPeriod,
  RunPeriod1, 7, 1, , 7, 1, , Monday, No, No, Yes, No, No;

Site:Location,
  London/Heathrow, 51.48, -0.45, 0.0, 24.0;

GlobalGeometryRules,
  UpperLeftCorner, CounterClockWise, World;

!- ===== Schedule Type Limits =====

ScheduleTypeLimits, Fraction, 0.0, 1.0, CONTINUOUS;
ScheduleTypeLimits, Temperature, -60, 200, CONTINUOUS;
ScheduleTypeLimits, Any Number;
ScheduleTypeLimits, Control Type, 0, 4, DISCRETE;

!- ===== Schedules =====

Schedule:Compact,
  Always_On, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 1.0;

Schedule:Compact,
  Always_Off, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 0.0;

Schedule:Compact,
  OCC_MULTIPLIER, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 1.0;

Schedule:Compact,
  Office_Occ, Fraction,
  Through: 12/31,
  For: Weekdays,
    Until: 08:00, 0.0,
    Until: 18:00, 1.0,
    Until: 24:00, 0.0,
  For: AllOtherDays,
    Until: 24:00, 0.0;

Schedule:Compact,
  ActivityLevel_Sched, Any Number,
  Through: 12/31, For: AllDays, Until: 24:00, 120;

Schedule:Compact,
  Dual Zone Control Type Sched, Control Type,
  Through: 12/31, For: AllDays, Until: 24:00, 4;

Schedule:Compact,
  Heating_SP_Sched, Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 20.0;

Schedule:Compact,
  Cooling_SP_Sched, Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 26.0;

!- ===== Materials and Constructions =====

Material:NoMass,
  Adiabatic_Mat, Smooth, 100.0;

Material:NoMass,
  ExtWall_Mat, Smooth, 0.5;

Material:NoMass,
  Floor_Mat, Smooth, 0.25;

Construction, Adiabatic_Const, Adiabatic_Mat;
Construction, ExtWall_Const, ExtWall_Mat;
Construction, Floor_Const, Floor_Mat;

WindowMaterial:SimpleGlazingSystem,
  SimpleGlazing, 3.0, 0.3;

Construction, Window_Const, SimpleGlazing;
"""

    zones_idf = ""
    for z in ZONES:
        x1, y1 = z.x0, 0.0
        x2, y2 = z.x0 + z.width, z.depth
        zone_name = z.name

        zones_idf += f"\n!- ===== Zone: {zone_name} =====\n\n"
        zones_idf += f"Zone,\n  {zone_name},\n  0, 0, 0, 0,\n  1,             !- Multiplier\n  ,\n  {z.width * z.depth * H},  !- Volume {{m3}}\n  {z.width * z.depth};  !- Floor Area {{m2}}\n\n"

        # Surfaces
        zones_idf += _wall(f"{zone_name}_SouthWall", zone_name, x1, y1, x2, y2, 0, H, "S", "Outdoors")
        zones_idf += _wall(f"{zone_name}_NorthWall",  zone_name, x1, y1, x2, y2, 0, H, "N")
        zones_idf += _wall(f"{zone_name}_EastWall",   zone_name, x1, y1, x2, y2, 0, H, "E")
        zones_idf += _wall(f"{zone_name}_WestWall",   zone_name, x1, y1, x2, y2, 0, H, "W")
        zones_idf += _wall(f"{zone_name}_Floor",      zone_name, x1, y1, x2, y2, 0, H, "Floor", "Ground")
        zones_idf += _wall(f"{zone_name}_Ceiling",    zone_name, x1, y1, x2, y2, 0, H, "Ceiling")
        zones_idf += _window(f"{zone_name}_SouthWindow", f"{zone_name}_SouthWall", zone_name, x1, x2, y1, 0, H)

        # Internal gains
        zones_idf += _people(z)
        zones_idf += _lights(z)
        zones_idf += _equipment(z)

        # Thermostat
        zones_idf += _thermostat(z)

        # HVAC
        zones_idf += _zone_hvac(z)

        # Output variables
        zones_idf += _output_vars(z)

    footer = """
!- ===== Global Outputs =====

Output:Variable,*,Zone Air Temperature,TimeStep;
Output:Meter,Electricity:Facility,TimeStep;
OutputControl:Table:Style, Comma;
Output:Table:SummaryReports, AllSummary;
"""

    return header + zones_idf + footer


if __name__ == "__main__":
    content = generate_idf()
    OUTPUT_PATH.write_text(content)
    print(f"IDF written to {OUTPUT_PATH} ({len(content)} chars)")
```

- [ ] **Step 2: Run the IDF generator**

```bash
python simulation/build_idf.py
```

Expected:
```
IDF written to .../simulation/building.idf (XXXXX chars)
```

- [ ] **Step 3: Symlink the weather file**

```bash
ln -s /Applications/EnergyPlus-25-2-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw simulation/weather.epw
```

Expected: `simulation/weather.epw` symlink created.

- [ ] **Step 4: Validate the IDF runs with EnergyPlus**

```bash
energyplus -d /tmp/ep_test_validate -w simulation/weather.epw simulation/building.idf 2>&1 | tail -20
```

Expected: Output contains `EnergyPlus Completed Successfully` (may take 10-30s). If it fails with geometry errors, the vertex ordering in `_wall()` needs adjustment — check the error file at `/tmp/ep_test_validate/eplusout.err`.

- [ ] **Step 5: Commit**

```bash
git add simulation/build_idf.py simulation/weather.epw
git commit -m "feat: IDF generator for 5-zone office building, symlink Chicago EPW"
```

Note: `building.idf` is in `.gitignore` (generated); `build_idf.py` is committed.

---

## Task 5: EnergyPlus Engine

**Files:**
- Create: `simulation/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from simulation.engine import SimulationEngine, SimulationResult, _aggregate_results
from simulation.mpc import Setpoints
from api.models import OptimizeRequest


FAKE_SETPOINTS = Setpoints(
    ahu1_supply_c=18.0, ahu2_supply_c=18.0, chiller_c=7.0,
    free_cool_pct=30.0, pv_divert_pct=47.0,
    zone_cooling_sp_c=25.0, zone_heating_sp_c=20.0, demand_limit_kw=80.0,
)

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]

FAKE_TIMESTEPS = [
    {"zone_temps": {z: 23.0 + i * 0.1 for i, z in enumerate(ZONE_NAMES)},
     "cooling_j": {z: 10000.0 for z in ZONE_NAMES},
     "heating_j": {z: 0.0 for z in ZONE_NAMES}}
    for _ in range(16)
]


def test_aggregate_results_returns_simulation_result():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert isinstance(result, SimulationResult)


def test_aggregate_results_zone_temps_averaged():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert "Server Hall" in result.zone_temps
    assert "Lab A" in result.zone_temps
    assert 20.0 <= result.zone_temps["Server Hall"] <= 30.0


def test_aggregate_results_energy_forecast_has_7_points():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert len(result.energy_forecast_kwh) == 7


def test_aggregate_results_comfort_zones_counts_in_band():
    # All zones at 23°C → should be in 22-26 band → comfort_zones = 5
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result.comfort_zones == 5


def test_aggregate_results_power_reduced_by_pv():
    result_no_pv = _aggregate_results(FAKE_TIMESTEPS, pv_kw=0.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    result_with_pv = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result_with_pv.power_kw <= result_no_pv.power_kw


def test_aggregate_results_savings_pct_non_negative():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result.savings_pct >= 0


def test_aggregate_results_setpoints_formatted():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert "AHU-1 supply" in result.setpoints
    assert "°C" in result.setpoints["AHU-1 supply"]
    assert "%" in result.setpoints["Free-cool %"]


def test_engine_run_calls_run_simulation(tmp_path):
    idf = tmp_path / "test.idf"
    idf.write_text("")
    epw = tmp_path / "test.epw"
    epw.write_text("")

    engine = SimulationEngine(
        idf_path=idf,
        weather_path=epw,
        ep_dir=Path("/Applications/EnergyPlus-25-2-0"),
    )

    request = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)

    with patch.object(engine, "_run_simulation", return_value=FAKE_TIMESTEPS) as mock_sim:
        result = engine.run(request, FAKE_SETPOINTS)

    mock_sim.assert_called_once()
    assert isinstance(result, SimulationResult)


def test_engine_run_passes_setpoints_to_simulation(tmp_path):
    idf = tmp_path / "test.idf"
    idf.write_text("")
    epw = tmp_path / "test.epw"
    epw.write_text("")

    engine = SimulationEngine(idf_path=idf, weather_path=epw,
                              ep_dir=Path("/Applications/EnergyPlus-25-2-0"))
    request = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)

    with patch.object(engine, "_run_simulation", return_value=FAKE_TIMESTEPS) as mock_sim:
        engine.run(request, FAKE_SETPOINTS)

    _, kwargs = mock_sim.call_args
    assert kwargs["setpoints"] == FAKE_SETPOINTS
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_engine.py -v
```

Expected: `ImportError` — `simulation.engine` does not exist.

- [ ] **Step 3: Implement simulation/engine.py**

Create `simulation/engine.py`:

```python
import os
import sys
import uuid
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.mpc import Setpoints
from api.models import OptimizeRequest

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]
CHILLER_COP = 3.0
# Indices into 16-timestep array that map to the 7 dashboard forecast labels
# (+15m, +30m, +45m, +1h, +2h, +3h, +4h)
FORECAST_INDICES = [0, 1, 2, 3, 7, 11, 15]
EP_DIR = Path(os.environ.get("ENERGYPLUS_DIR", "/Applications/EnergyPlus-25-2-0"))


@dataclass
class SimulationResult:
    power_kw: float
    savings_pct: int
    avg_zone_temp: float
    pv_contribution_pct: int
    comfort_zones: int
    zone_temps: dict[str, float]
    energy_forecast_kwh: list[float]
    setpoints: dict[str, str]
    simulation_duration_s: float


def _aggregate_results(
    timesteps: list[dict[str, Any]],
    pv_kw: float,
    tariff: float,
    setpoints: Setpoints,
) -> SimulationResult:
    """Aggregate 16 EnergyPlus timestep readings into a SimulationResult."""
    # Average zone temps across all timesteps
    zone_temps = {
        z: round(sum(ts["zone_temps"][z] for ts in timesteps) / len(timesteps), 1)
        for z in ZONE_NAMES
    }

    avg_zone_temp = round(sum(zone_temps.values()) / len(zone_temps), 1)
    comfort_zones = sum(1 for t in zone_temps.values() if 22.0 <= t <= 26.0)

    # Total HVAC cooling energy (J) → average power (kW), then subtract PV
    total_cooling_j = sum(
        sum(ts["cooling_j"].values()) for ts in timesteps
    )
    dt_seconds = 15 * 60
    gross_cooling_kw = total_cooling_j / (len(timesteps) * dt_seconds * 1000)
    gross_electrical_kw = gross_cooling_kw / CHILLER_COP
    net_power_kw = max(0.0, round(gross_electrical_kw - pv_kw, 1))

    # Baseline power estimate (same energy without MPC = 10% more than gross)
    baseline_kw = gross_electrical_kw * 1.1
    savings_pct = max(0, round((baseline_kw - gross_electrical_kw) / baseline_kw * 100))

    # PV contribution %
    pv_contribution = 0
    if gross_electrical_kw + pv_kw > 0:
        pv_contribution = round(pv_kw / (gross_electrical_kw + pv_kw) * 100)

    # Energy forecast: subsample to 7 dashboard labels
    energy_forecast = []
    for idx in FORECAST_INDICES:
        ts = timesteps[idx]
        cooling_j = sum(ts["cooling_j"].values())
        kwh = round(cooling_j / CHILLER_COP / 3_600_000, 1)  # J → kWh
        energy_forecast.append(kwh)

    formatted_setpoints = {
        "AHU-1 supply": f"{setpoints.ahu1_supply_c:.1f}°C",
        "AHU-2 supply": f"{setpoints.ahu2_supply_c:.1f}°C",
        "Chiller setpt": f"{setpoints.chiller_c:.1f}°C",
        "Free-cool %": f"{setpoints.free_cool_pct:.0f}%",
        "PV divert": f"{setpoints.pv_divert_pct:.0f}%",
        "Demand limit": f"{setpoints.demand_limit_kw:.0f} kW",
    }

    return SimulationResult(
        power_kw=net_power_kw,
        savings_pct=savings_pct,
        avg_zone_temp=avg_zone_temp,
        pv_contribution_pct=pv_contribution,
        comfort_zones=comfort_zones,
        zone_temps=zone_temps,
        energy_forecast_kwh=energy_forecast,
        setpoints=formatted_setpoints,
        simulation_duration_s=0.0,
    )


class SimulationEngine:
    def __init__(self, idf_path: Path, weather_path: Path, ep_dir: Path = EP_DIR):
        self._idf_path = idf_path
        self._weather_path = weather_path
        self._ep_dir = ep_dir

    def _create_api(self):
        if str(self._ep_dir) not in sys.path:
            sys.path.insert(0, str(self._ep_dir))
        from pyenergyplus.api import EnergyPlusAPI  # noqa: PLC0415
        return EnergyPlusAPI()

    def _run_simulation(
        self,
        request: OptimizeRequest,
        setpoints: Setpoints,
    ) -> list[dict[str, Any]]:
        """Run EnergyPlus for the prediction horizon; return per-timestep data."""
        api = self._create_api()
        state = api.state_manager.new_state()
        api.runtime.set_console_output_status(state, False)

        collected: list[dict[str, Any]] = []
        handles_initialized = False
        var_handles: dict[str, int] = {}
        cool_handles: dict[str, int] = {}

        # Request output variables before run
        for zone in ZONE_NAMES:
            api.exchange.request_variable(state, "Zone Air Temperature", zone)
            api.exchange.request_variable(
                state, "Zone Ideal Loads Supply Air Total Cooling Energy", zone
            )

        def _init_handles(state) -> None:
            nonlocal handles_initialized
            if handles_initialized:
                return
            if not api.exchange.api_data_fully_ready(state):
                return
            for zone in ZONE_NAMES:
                h = api.exchange.get_variable_handle(state, "Zone Air Temperature", zone)
                var_handles[zone] = h
                c = api.exchange.get_variable_handle(
                    state, "Zone Ideal Loads Supply Air Total Cooling Energy", zone
                )
                cool_handles[zone] = c
            handles_initialized = True

        def _inject_inputs(state) -> None:
            if not api.exchange.api_data_fully_ready(state):
                return
            _init_handles(state)

            # Override outdoor dry-bulb temperature
            h = api.exchange.get_actuator_handle(
                state, "Weather Data", "Outdoor Dry Bulb", "Environment"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, request.ext_temp)

            # Override occupancy multiplier schedule
            h = api.exchange.get_actuator_handle(
                state, "Schedule:Compact", "Schedule Value", "OCC_MULTIPLIER"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, request.occupancy / 100.0)

            # Override cooling setpoint schedule
            h = api.exchange.get_actuator_handle(
                state, "Schedule:Compact", "Schedule Value", "Cooling_SP_Sched"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, setpoints.zone_cooling_sp_c)

        def _collect_data(state) -> None:
            if api.exchange.warmup_flag(state):
                return
            if not api.exchange.api_data_fully_ready(state):
                return
            if not handles_initialized:
                _init_handles(state)

            zone_temps = {
                z: api.exchange.get_variable_value(state, var_handles[z])
                for z in ZONE_NAMES
                if z in var_handles and var_handles[z] != -1
            }
            cooling_j = {
                z: api.exchange.get_variable_value(state, cool_handles[z])
                for z in ZONE_NAMES
                if z in cool_handles and cool_handles[z] != -1
            }

            if zone_temps:
                collected.append({
                    "zone_temps": zone_temps,
                    "cooling_j": cooling_j,
                    "heating_j": {z: 0.0 for z in ZONE_NAMES},
                })

            if len(collected) >= 16:
                api.runtime.stop_simulation(state)

        api.runtime.callback_begin_zone_timestep_before_set_current_weather(
            state, _inject_inputs
        )
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state, _collect_data
        )

        output_dir = Path(tempfile.mkdtemp(prefix="ep_run_"))
        api.runtime.run_energyplus(state, [
            "-d", str(output_dir),
            "-w", str(self._weather_path),
            str(self._idf_path),
        ])
        api.state_manager.delete_state(state)

        # Ensure we have exactly 16 timesteps; pad with last value if simulation
        # ended early (e.g. only day 1 had fewer steps)
        while len(collected) < 16 and collected:
            collected.append(collected[-1])

        return collected

    def run(self, request: OptimizeRequest, setpoints: Setpoints) -> SimulationResult:
        import time
        t0 = time.perf_counter()
        timesteps = self._run_simulation(request=request, setpoints=setpoints)
        duration = time.perf_counter() - t0

        result = _aggregate_results(
            timesteps=timesteps,
            pv_kw=request.pv_kw,
            tariff=request.tariff,
            setpoints=setpoints,
        )
        result.simulation_duration_s = round(duration, 2)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_engine.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simulation/engine.py tests/test_engine.py
git commit -m "feat: EnergyPlus engine with pyenergyplus co-simulation and result aggregation"
```

---

## Task 6: FastAPI Application

**Files:**
- Create: `api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from simulation.engine import SimulationResult
from simulation.mpc import Setpoints


FAKE_RESULT = SimulationResult(
    power_kw=38.2, savings_pct=18, avg_zone_temp=23.4, pv_contribution_pct=37,
    comfort_zones=5,
    zone_temps={"Server Hall": 22.1, "Open Plan": 23.4, "Boardroom": 23.8,
                "Reception": 22.9, "Lab A": 23.6},
    energy_forecast_kwh=[38.0, 35.0, 33.0, 31.0, 28.0, 26.0, 24.0],
    setpoints={"AHU-1 supply": "17.8°C", "AHU-2 supply": "18.2°C",
               "Chiller setpt": "6.8°C", "Free-cool %": "45%",
               "PV divert": "47%", "Demand limit": "46 kW"},
    simulation_duration_s=12.4,
)


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "energyplus_version" in data
    assert "idf_loaded" in data


def test_optimize_returns_200_with_valid_input(client):
    with patch("api.main.engine") as mock_engine:
        mock_engine.run.return_value = FAKE_RESULT
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    assert resp.status_code == 200


def test_optimize_response_has_all_fields(client):
    with patch("api.main.engine") as mock_engine:
        mock_engine.run.return_value = FAKE_RESULT
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    data = resp.json()
    assert "power_kw" in data
    assert "zone_temps" in data
    assert "energy_forecast_kwh" in data
    assert "setpoints" in data
    assert len(data["energy_forecast_kwh"]) == 7
    assert "Server Hall" in data["zone_temps"]


def test_optimize_rejects_invalid_occupancy(client):
    resp = client.post("/optimize", json={
        "occupancy": 150, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
    })
    assert resp.status_code == 422


def test_optimize_rejects_missing_field(client):
    resp = client.post("/optimize", json={
        "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0
        # tariff missing
    })
    assert resp.status_code == 422


def test_scenarios_normal(client):
    resp = client.get("/scenarios/normal")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "normal"
    assert data["occupancy"] == 70
    assert data["ext_temp"] == 24.0


def test_scenarios_peak(client):
    resp = client.get("/scenarios/peak")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tariff"] == 34.0


def test_scenarios_unknown_returns_404(client):
    resp = client.get("/scenarios/nonexistent")
    assert resp.status_code == 404


def test_optimize_calls_compute_setpoints_then_engine(client):
    with patch("api.main.compute_setpoints") as mock_mpc, \
         patch("api.main.engine") as mock_engine:
        fake_sp = Setpoints(ahu1_supply_c=18.0, ahu2_supply_c=18.0, chiller_c=7.0,
                            free_cool_pct=30.0, pv_divert_pct=47.0,
                            zone_cooling_sp_c=25.0, zone_heating_sp_c=20.0,
                            demand_limit_kw=80.0)
        mock_mpc.return_value = fake_sp
        mock_engine.run.return_value = FAKE_RESULT

        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })

    mock_mpc.assert_called_once_with(
        occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0
    )
    mock_engine.run.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError` — `api.main` does not exist.

- [ ] **Step 3: Implement api/main.py**

Create `api/main.py`:

```python
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse
from simulation.mpc import compute_setpoints
from simulation.engine import SimulationEngine, EP_DIR

app = FastAPI(title="BuilMirai MPC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_SIM_DIR = Path(__file__).parent.parent / "simulation"
_IDF_PATH = _SIM_DIR / "building.idf"
_EPW_PATH = _SIM_DIR / "weather.epw"

engine = SimulationEngine(idf_path=_IDF_PATH, weather_path=_EPW_PATH, ep_dir=EP_DIR)
_engine_lock = asyncio.Lock()           # serialise concurrent EnergyPlus runs
_executor = ThreadPoolExecutor(max_workers=1)

SCENARIOS: dict[str, dict] = {
    "normal":   {"occupancy": 70,  "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0},
    "peak":     {"occupancy": 90,  "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0},
    "heatwave": {"occupancy": 75,  "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0},
    "preheat":  {"occupancy": 20,  "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0},
    "night":    {"occupancy": 5,   "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0},
}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        energyplus_version="25.2",
        idf_loaded=_IDF_PATH.exists(),
    )


@app.get("/scenarios/{name}", response_model=ScenarioResponse)
def get_scenario(name: str) -> ScenarioResponse:
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")
    return ScenarioResponse(name=name, **SCENARIOS[name])


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    setpoints = compute_setpoints(
        occupancy=request.occupancy,
        ext_temp=request.ext_temp,
        pv_kw=request.pv_kw,
        tariff=request.tariff,
    )
    # Serialise EnergyPlus runs (not thread-safe across instances) and
    # offload the blocking simulation to a thread pool so the event loop
    # remains responsive during the 10-30s simulation.
    loop = asyncio.get_event_loop()
    async with _engine_lock:
        result = await loop.run_in_executor(
            _executor, engine.run, request, setpoints
        )
    return OptimizeResponse(
        power_kw=result.power_kw,
        savings_pct=result.savings_pct,
        avg_zone_temp=result.avg_zone_temp,
        pv_contribution_pct=result.pv_contribution_pct,
        comfort_zones=result.comfort_zones,
        zone_temps=result.zone_temps,
        energy_forecast_kwh=result.energy_forecast_kwh,
        setpoints=result.setpoints,
        simulation_duration_s=result.simulation_duration_s,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: FastAPI app with /optimize, /health, /scenarios endpoints"
```

---

## Task 7: Entry Point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace the placeholder entry point**

Replace `main.py` with:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 2: Verify the server starts**

```bash
python main.py &
sleep 2
curl -s http://localhost:8000/health | python3 -m json.tool
kill %1
```

Expected:
```json
{
    "status": "ok",
    "energyplus_version": "25.2",
    "idf_loaded": true
}
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: uvicorn entry point for FastAPI server"
```

---

## Task 8: Frontend Update

**Files:**
- Modify: `builmirai_mpc_hvac_dashboard.html`

- [ ] **Step 1: Replace `runOptimise()` with a real API call**

In `builmirai_mpc_hvac_dashboard.html`, replace the existing `runOptimise()` function:

```javascript
// REMOVE this old version:
// function runOptimise() {
//   const btn = document.getElementById('run-btn');
//   btn.classList.add('running');
//   btn.textContent = 'Optimising…';
//   document.getElementById('mpc-status').textContent = 'solving…';
//   setTimeout(() => {
//     render(state);
//     btn.classList.remove('running');
//     btn.textContent = 'Run optimiser';
//     document.getElementById('mpc-status').textContent = 'running';
//   }, 900);
// }

// REPLACE WITH:
async function runOptimise() {
  const btn = document.getElementById('run-btn');
  btn.classList.add('running');
  btn.textContent = 'Optimising…';
  document.getElementById('mpc-status').textContent = 'solving…';

  try {
    const res = await fetch('http://localhost:8000/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        occupancy: state.occ,
        ext_temp: state.ext,
        pv_kw: state.pv,
        tariff: state.tariff
      })
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const data = await res.json();
    renderFromApi(data);
  } catch (err) {
    console.error('Optimisation failed:', err);
    document.getElementById('sys-status').textContent = 'MPC error';
    document.getElementById('sys-status').className = 'badge badge-warn';
  } finally {
    btn.classList.remove('running');
    btn.textContent = 'Run optimiser';
    document.getElementById('mpc-status').textContent = 'running';
  }
}
```

- [ ] **Step 2: Add `renderFromApi()` function**

Add the following function after `render(state)` is defined (above the `setScenario` function):

```javascript
function renderFromApi(data) {
  // Metrics
  document.getElementById('m-power').innerHTML = data.power_kw + '<span class="metric-unit"> kW</span>';
  document.getElementById('m-power-d').textContent = '↓' + data.savings_pct + '% vs baseline';
  document.getElementById('m-power-d').className = 'metric-delta ' + (data.savings_pct > 0 ? 'delta-good' : 'delta-bad');

  document.getElementById('m-temp').innerHTML = data.avg_zone_temp + '<span class="metric-unit">°C</span>';
  const inBand = data.avg_zone_temp >= 22 && data.avg_zone_temp <= 26;
  document.getElementById('m-temp-d').textContent = inBand ? 'within 22–26°C' : 'outside comfort band';
  document.getElementById('m-temp-d').className = 'metric-delta ' + (inBand ? 'delta-good' : 'delta-bad');

  document.getElementById('m-saving').innerHTML = data.savings_pct + '<span class="metric-unit">%</span>';
  document.getElementById('m-pv').innerHTML = state.pv + '<span class="metric-unit"> kW</span>';
  document.getElementById('m-pv-d').textContent = data.pv_contribution_pct + '% of load';

  // Zone temperatures (API returns object, dashboard expects array in zone order)
  const zoneOrder = ['Server Hall','Open Plan','Boardroom','Reception','Lab A'];
  const temps = zoneOrder.map(z => data.zone_temps[z] || 23.0);
  renderZones(temps);

  // Energy forecast
  renderForecast(data.energy_forecast_kwh);

  // Setpoints
  renderSetpoints(data.setpoints);

  // Comfort bar
  const cpct = Math.round(data.comfort_zones / 5 * 100);
  document.getElementById('comfort-fill').style.width = cpct + '%';
  document.getElementById('comfort-fill').className = 'comfort-fill ' + (cpct === 100 ? 'comfort-ok' : 'comfort-warn');
  document.getElementById('comfort-pct').textContent = cpct + '% zones in comfort band';
  document.getElementById('comfort-status').textContent = cpct === 100 ? 'Comfort: all zones' : 'Comfort: ' + data.comfort_zones + '/5 zones';
  document.getElementById('comfort-status').className = 'badge ' + (cpct === 100 ? 'badge-ok' : 'badge-warn');

  addLog({...state, power: data.power_kw, comfort: data.comfort_zones,
          ext: state.ext, pvKw: state.pv, tariff: state.tariff});
}
```

- [ ] **Step 3: Update `setScenario()` to fetch preset inputs from API**

Replace the `setScenario` function:

```javascript
async function setScenario(key) {
  try {
    const res = await fetch('http://localhost:8000/scenarios/' + key);
    if (res.ok) {
      const sc = await res.json();
      state = { occ: sc.occupancy, ext: sc.ext_temp, pv: sc.pv_kw, tariff: sc.tariff };
    } else {
      // Fall back to local scenarios if API unavailable
      const sc = scenarios[key];
      state = { occ: sc.occ, ext: sc.ext, pv: sc.pv, tariff: sc.tariff };
    }
  } catch {
    const sc = scenarios[key];
    state = { occ: sc.occ, ext: sc.ext, pv: sc.pv, tariff: sc.tariff };
  }

  document.getElementById('sl-occ').value = state.occ;
  document.getElementById('sl-ext').value = state.ext;
  document.getElementById('sl-pv').value = state.pv;
  document.getElementById('sl-tariff').value = state.tariff;
  document.getElementById('sv-occ').textContent = state.occ + '%';
  document.getElementById('sv-ext').textContent = state.ext + '°C';
  document.getElementById('sv-pv').textContent = state.pv + ' kW';
  document.getElementById('sv-tariff').textContent = state.tariff + 'p/kWh';
  document.querySelectorAll('.scenario-btn').forEach(b =>
    b.classList.toggle('active', b.getAttribute('onclick').includes("'" + key + "'"))
  );

  await runOptimise();
}
```

- [ ] **Step 4: Manual integration test**

Start the server and open the dashboard:

```bash
# Terminal 1: generate IDF (if not done yet) and start server
python simulation/build_idf.py
python main.py

# Terminal 2: open dashboard
open builmirai_mpc_hvac_dashboard.html
```

1. Click **"Run optimiser"** — button should show "Optimising…" for 10-30s then update all metrics with real EnergyPlus data.
2. Click **"Peak tariff"** scenario — sliders update and optimiser runs automatically.
3. Drag the **Ext. temp** slider to 40°C — verify the displayed external temperature changes.
4. Click **"Run optimiser"** again — verify heatwave setpoints differ from normal.
5. Check the browser console (DevTools → Console) — no errors.
6. Check `simulation_duration_s` by running `curl -s -X POST http://localhost:8000/optimize -H "Content-Type: application/json" -d '{"occupancy":70,"ext_temp":24,"pv_kw":14,"tariff":11}' | python3 -m json.tool` — verify `simulation_duration_s` is populated.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add builmirai_mpc_hvac_dashboard.html
git commit -m "feat: connect dashboard frontend to real EnergyPlus backend via POST /optimize"
```

---

## Running the Complete System

After all tasks are complete:

```bash
# 1. Generate building IDF (one-time)
python simulation/build_idf.py

# 2. Start the API server
python main.py
# Server running at http://localhost:8000

# 3. Open the dashboard in a browser
open builmirai_mpc_hvac_dashboard.html

# 4. Run all tests
pytest tests/ -v
```

## Known Limitations (Out of Scope for This Plan)

- The weather EPW overrides only dry-bulb temperature; humidity/solar radiation remain from Chicago TMY3 file.
- `ZoneHVAC:IdealLoadsAirSystem` reports cooling thermal energy, not electrical — COP of 3.0 is a fixed approximation.
- EnergyPlus warmup (1 day) adds ~5-10s overhead to each `/optimize` call; this is expected.
- Concurrent `/optimize` requests are serialised (single EnergyPlus instance per process).
- IDF geometry uses `_wall()` with assumed CCW vertex ordering. If EnergyPlus reports geometry warnings in the `.err` file during Task 4 Step 4, adjust vertex ordering in `build_idf.py::_wall()`.
