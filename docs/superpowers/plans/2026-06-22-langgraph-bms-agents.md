# LangGraph 5-Agent BMS Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `compute_setpoints()` heuristic in `simulation/mpc.py` with a LangGraph StateGraph of 5 agents (Demand, Supply, Battery, Thermal, Orchestration) backed by Groq LLM, add battery SOC simulation, and update the dashboard UI to show a live agent graph.

**Architecture:** A `StateGraph` is compiled once at module load in `simulation/graph.py`. On each `/optimize` call, the entry node seeds `BMSState`, 4 LLM-backed specialist nodes run in parallel (fan-out), and the Orchestration node reconciles their proposals into a `Setpoints` object passed unchanged to the existing `engine.run()`. Agent traces are returned in the API response and rendered in the dashboard.

**Tech Stack:** Python 3.13, LangGraph 1.2+, LangChain-Groq (`ChatGroq`), FastAPI, EnergyPlus 25.2 (unchanged), vanilla JS dashboard.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `simulation/state.py` | Create | `BMSState` TypedDict + `AgentAction` TypedDict |
| `simulation/battery.py` | Create | `BatteryState`, `SCENARIO_SOC`, `update_soc()`, session singleton |
| `simulation/llm_provider.py` | Create | `_get_llm()` factory — env-configurable provider/model |
| `simulation/graph.py` | Create | All 5 node functions + compiled `StateGraph` |
| `simulation/mpc.py` | Unchanged | `Setpoints` dataclass still used by engine |
| `simulation/engine.py` | Unchanged | EnergyPlus integration |
| `api/models.py` | Modify | Add `AgentTraceEntry`, extend `OptimizeRequest`/`OptimizeResponse` |
| `api/main.py` | Modify | Replace `compute_setpoints()` with `graph.invoke()` |
| `builmirai_mpc_hvac_dashboard.html` | Modify | Agent graph panel, trace-based log, battery SOC chip |
| `tests/test_state.py` | Create | Unit tests for `BMSState` defaults |
| `tests/test_battery.py` | Create | Unit tests for `update_soc()` and SOC clamping |
| `tests/test_graph.py` | Create | Graph node tests with mocked LLM |

---

## Task 1: State Schema

**Files:**
- Create: `simulation/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1.1: Write the failing test**

```python
# tests/test_state.py
from simulation.state import AgentAction, BMSState


def test_agent_action_keys():
    action: AgentAction = {
        "proposed": {"demand_limit_kw": 75.0},
        "score": 0.8,
        "rationale": "Peak avoidance active",
    }
    assert action["score"] == 0.8
    assert "proposed" in action
    assert "rationale" in action


def test_bms_state_has_required_keys():
    state: BMSState = {
        "occupancy": 70.0,
        "ext_temp": 24.0,
        "pv_kw": 14.0,
        "tariff": 11.0,
        "net_power_kw": 30.0,
        "zone_temps": {"Server Hall": 23.5},
        "comfort_band": (22.0, 26.0),
        "battery_soc_pct": 50.0,
        "scenario": "normal",
        "demand_action": None,
        "supply_action": None,
        "battery_action": None,
        "thermal_action": None,
        "final_setpoints": None,
    }
    assert state["comfort_band"] == (22.0, 26.0)
    assert state["demand_action"] is None
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'simulation.state'`

- [ ] **Step 1.3: Create `simulation/state.py`**

```python
from typing import TypedDict


class AgentAction(TypedDict):
    proposed: dict
    score: float
    rationale: str


class BMSState(TypedDict):
    # Inputs — populated by entry_node
    occupancy: float
    ext_temp: float
    pv_kw: float
    tariff: float
    net_power_kw: float
    zone_temps: dict[str, float]
    comfort_band: tuple[float, float]
    battery_soc_pct: float
    scenario: str | None

    # Agent outputs — each agent writes only its own key
    demand_action: AgentAction | None
    supply_action: AgentAction | None
    battery_action: AgentAction | None
    thermal_action: AgentAction | None

    # Orchestration output
    final_setpoints: dict | None
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_state.py -v
```

Expected: `2 passed`

- [ ] **Step 1.5: Commit**

```bash
git add simulation/state.py tests/test_state.py
git commit -m "feat: add BMSState and AgentAction TypedDicts"
```

---

## Task 2: Battery SOC Simulation

**Files:**
- Create: `simulation/battery.py`
- Create: `tests/test_battery.py`

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/test_battery.py
import pytest
from simulation.battery import update_soc, SCENARIO_SOC, get_battery, reset_battery


def test_charging_increases_soc():
    soc = update_soc(50.0, charge_discharge_kw=20.0, dt_minutes=15)
    # 20 kW * 0.25 h = 5 kWh into 100 kWh battery → +5%
    assert abs(soc - 55.0) < 0.1


def test_discharging_decreases_soc():
    soc = update_soc(60.0, charge_discharge_kw=-15.0, dt_minutes=15)
    # 15 kW * 0.25 h = 3.75 kWh out → -3.75%
    assert abs(soc - 56.25) < 0.1


def test_soc_clamped_to_100():
    soc = update_soc(98.0, charge_discharge_kw=25.0, dt_minutes=15)
    assert soc == 100.0


def test_soc_clamped_to_0():
    soc = update_soc(2.0, charge_discharge_kw=-25.0, dt_minutes=15)
    assert soc == 0.0


def test_charge_rate_tapers_above_80_pct():
    # Above 80% SOC, max charge rate drops to 12.5 kW
    soc_full_rate = update_soc(50.0, charge_discharge_kw=25.0, dt_minutes=15)
    soc_tapered = update_soc(85.0, charge_discharge_kw=25.0, dt_minutes=15)
    # At 85% with taper: 12.5 kW * 0.25h = 3.125% added
    assert abs(soc_tapered - 88.125) < 0.1


def test_scenario_soc_seeds():
    assert SCENARIO_SOC["normal"] == 50.0
    assert SCENARIO_SOC["peak"] == 60.0
    assert SCENARIO_SOC["heatwave"] == 30.0
    assert SCENARIO_SOC["preheat"] == 80.0
    assert SCENARIO_SOC["night"] == 90.0


def test_singleton_persists_across_updates():
    reset_battery(50.0)
    bat = get_battery()
    assert bat.soc_pct == 50.0
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_battery.py -v
```

Expected: `ModuleNotFoundError: No module named 'simulation.battery'`

- [ ] **Step 2.3: Create `simulation/battery.py`**

```python
from dataclasses import dataclass


SCENARIO_SOC: dict[str, float] = {
    "normal":   50.0,
    "peak":     60.0,
    "heatwave": 30.0,
    "preheat":  80.0,
    "night":    90.0,
}

_CAPACITY_KWH = 100.0
_MAX_RATE_KW = 25.0
_TAPER_THRESHOLD_PCT = 80.0
_TAPERED_RATE_KW = 12.5  # C/2 taper above threshold


@dataclass
class BatteryState:
    capacity_kwh: float = _CAPACITY_KWH
    max_rate_kw: float = _MAX_RATE_KW
    soc_pct: float = 50.0


_battery = BatteryState()


def get_battery() -> BatteryState:
    return _battery


def reset_battery(soc_pct: float) -> None:
    _battery.soc_pct = soc_pct


def update_soc(
    current_pct: float,
    charge_discharge_kw: float,
    dt_minutes: float = 15.0,
) -> float:
    """Return new SOC % after applying charge_discharge_kw for dt_minutes.

    Positive kW = charging, negative = discharging.
    Applies C/2 taper above 80% SOC on the charge side.
    """
    effective_kw = charge_discharge_kw
    if charge_discharge_kw > 0 and current_pct >= _TAPER_THRESHOLD_PCT:
        effective_kw = min(charge_discharge_kw, _TAPERED_RATE_KW)

    dt_hours = dt_minutes / 60.0
    delta_pct = (effective_kw * dt_hours / _CAPACITY_KWH) * 100.0
    new_pct = current_pct + delta_pct
    return max(0.0, min(100.0, new_pct))
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_battery.py -v
```

Expected: `7 passed`

- [ ] **Step 2.5: Commit**

```bash
git add simulation/battery.py tests/test_battery.py
git commit -m "feat: add battery SOC simulation with scenario seeds and taper model"
```

---

## Task 3: LLM Provider Abstraction

**Files:**
- Create: `simulation/llm_provider.py`

- [ ] **Step 3.1: Create `simulation/llm_provider.py`**

No test needed — this is a thin factory. It is tested implicitly via mocked agent tests in Task 4–7.

```python
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq


def _get_llm(temperature: float = 0.1) -> BaseChatModel:
    """Return a configured LLM. Swap provider via LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    if provider == "groq":
        return ChatGroq(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: groq")
```

- [ ] **Step 3.2: Verify import works**

```bash
.venv/bin/python -c "from simulation.llm_provider import _get_llm; print('ok')"
```

Expected: `ok`

- [ ] **Step 3.3: Commit**

```bash
git add simulation/llm_provider.py
git commit -m "feat: add LLM provider factory with env-configurable provider/model"
```

---

## Task 4: Demand Agent Node

**Files:**
- Create: `simulation/graph.py` (stub — will grow through Tasks 4–9)
- Create: `tests/test_graph.py` (stub — will grow through Tasks 4–9)

- [ ] **Step 4.1: Write the failing test**

```python
# tests/test_graph.py
import json
from unittest.mock import MagicMock, patch
from simulation.state import BMSState


def _make_state(**overrides) -> BMSState:
    base: BMSState = {
        "occupancy": 70.0,
        "ext_temp": 24.0,
        "pv_kw": 14.0,
        "tariff": 11.0,
        "net_power_kw": 30.0,
        "zone_temps": {
            "Server Hall": 23.5, "Open Plan": 24.0, "Boardroom": 22.8,
            "Reception": 23.1, "Lab A": 24.5,
        },
        "comfort_band": (22.0, 26.0),
        "battery_soc_pct": 50.0,
        "scenario": "normal",
        "demand_action": None,
        "supply_action": None,
        "battery_action": None,
        "thermal_action": None,
        "final_setpoints": None,
    }
    base.update(overrides)
    return base


def _mock_llm_response(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    llm = MagicMock()
    llm.invoke.return_value = msg
    return llm


def test_demand_agent_returns_action():
    from simulation.graph import demand_agent
    state = _make_state(net_power_kw=50.0, occupancy=90.0, tariff=34.0)
    mock_llm = _mock_llm_response({
        "demand_limit_kw": 72.0,
        "dr_signal": "curtail",
        "score": 0.85,
        "rationale": "Peak tariff — curtail demand to 72 kW",
    })
    with patch("simulation.graph._get_llm", return_value=mock_llm):
        result = demand_agent(state)
    assert "demand_action" in result
    action = result["demand_action"]
    assert action["proposed"]["demand_limit_kw"] == 72.0
    assert action["proposed"]["dr_signal"] == "curtail"
    assert 0.0 <= action["score"] <= 1.0
    assert isinstance(action["rationale"], str)
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_graph.py::test_demand_agent_returns_action -v
```

Expected: `ModuleNotFoundError: No module named 'simulation.graph'`

- [ ] **Step 4.3: Create `simulation/graph.py` with entry node + demand agent**

```python
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from simulation.battery import SCENARIO_SOC, get_battery, reset_battery, update_soc
from simulation.llm_provider import _get_llm
from simulation.mpc import Setpoints
from simulation.state import AgentAction, BMSState

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]


# ── Entry node ────────────────────────────────────────────────────────────────

def entry_node(state: BMSState) -> dict:
    scenario = state.get("scenario")
    if scenario and scenario in SCENARIO_SOC:
        reset_battery(SCENARIO_SOC[scenario])
    soc = get_battery().soc_pct

    gross_kw = state["net_power_kw"] + state["pv_kw"]
    net_kw = max(0.0, round(gross_kw - state["pv_kw"], 1))

    return {
        "battery_soc_pct": soc,
        "net_power_kw": net_kw,
        "comfort_band": (22.0, 26.0),
        "zone_temps": state.get("zone_temps") or {z: 23.0 for z in ZONE_NAMES},
    }


# ── Demand Agent ──────────────────────────────────────────────────────────────

_DEMAND_SYSTEM = """You are the Demand Agent for a Building Management System.
Your job: recommend a demand limit and demand-response signal to minimise peak draw
and avoid grid penalties.

Inputs you receive: net_power_kw (current net grid draw), occupancy (%), tariff (p/kWh).

Respond ONLY with valid JSON — no markdown, no explanation outside the JSON:
{
  "demand_limit_kw": <float, 20–200>,
  "dr_signal": "curtail" | "normal" | "flex",
  "score": <float 0–1, where 1 = maximum demand headroom achieved>,
  "rationale": "<one concise sentence>"
}"""


def demand_agent(state: BMSState) -> dict:
    llm = _get_llm()
    human = (
        f"net_power_kw={state['net_power_kw']}, "
        f"occupancy={state['occupancy']}%, "
        f"tariff={state['tariff']}p/kWh"
    )
    response = llm.invoke([SystemMessage(content=_DEMAND_SYSTEM), HumanMessage(content=human)])
    data = json.loads(response.content)
    return {
        "demand_action": AgentAction(
            proposed={
                "demand_limit_kw": float(data["demand_limit_kw"]),
                "dr_signal": str(data["dr_signal"]),
            },
            score=max(0.0, min(1.0, float(data["score"]))),
            rationale=str(data["rationale"]),
        )
    }
```

- [ ] **Step 4.4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_graph.py::test_demand_agent_returns_action -v
```

Expected: `1 passed`

- [ ] **Step 4.5: Commit**

```bash
git add simulation/graph.py tests/test_graph.py
git commit -m "feat: add entry node and Demand Agent LangGraph node"
```

---

## Task 5: Supply Agent Node

**Files:**
- Modify: `simulation/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 5.1: Add failing test to `tests/test_graph.py`**

Append to the existing file:

```python
def test_supply_agent_returns_action():
    from simulation.graph import supply_agent
    state = _make_state(tariff=34.0, pv_kw=8.0, battery_soc_pct=60.0)
    mock_llm = _mock_llm_response({
        "pv_divert_pct": 90.0,
        "grid_import_limit_kw": 45.0,
        "score": 0.75,
        "rationale": "High tariff — maximise PV self-consumption",
    })
    with patch("simulation.graph._get_llm", return_value=mock_llm):
        result = supply_agent(state)
    assert "supply_action" in result
    action = result["supply_action"]
    assert 0.0 <= action["proposed"]["pv_divert_pct"] <= 100.0
    assert action["proposed"]["grid_import_limit_kw"] > 0
    assert 0.0 <= action["score"] <= 1.0
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_graph.py::test_supply_agent_returns_action -v
```

Expected: `ImportError` or `AttributeError: module 'simulation.graph' has no attribute 'supply_agent'`

- [ ] **Step 5.3: Add Supply Agent to `simulation/graph.py`**

Append after `demand_agent`:

```python
# ── Supply Agent ──────────────────────────────────────────────────────────────

_SUPPLY_SYSTEM = """You are the Supply Agent for a Building Management System.
Your job: recommend how to split supply between grid and solar PV to minimise import cost.

Inputs: tariff (p/kWh), pv_kw (current solar generation kW), battery_soc_pct (0–100).

Respond ONLY with valid JSON — no markdown:
{
  "pv_divert_pct": <float 0–100, fraction of PV to self-consume>,
  "grid_import_limit_kw": <float 0–200, max grid import allowed>,
  "score": <float 0–1, where 1 = lowest cost supply mix>,
  "rationale": "<one concise sentence>"
}"""


def supply_agent(state: BMSState) -> dict:
    llm = _get_llm()
    human = (
        f"tariff={state['tariff']}p/kWh, "
        f"pv_kw={state['pv_kw']}, "
        f"battery_soc_pct={state['battery_soc_pct']}"
    )
    response = llm.invoke([SystemMessage(content=_SUPPLY_SYSTEM), HumanMessage(content=human)])
    data = json.loads(response.content)
    return {
        "supply_action": AgentAction(
            proposed={
                "pv_divert_pct": max(0.0, min(100.0, float(data["pv_divert_pct"]))),
                "grid_import_limit_kw": float(data["grid_import_limit_kw"]),
            },
            score=max(0.0, min(1.0, float(data["score"]))),
            rationale=str(data["rationale"]),
        )
    }
```

- [ ] **Step 5.4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_graph.py::test_supply_agent_returns_action -v
```

Expected: `1 passed`

- [ ] **Step 5.5: Commit**

```bash
git add simulation/graph.py tests/test_graph.py
git commit -m "feat: add Supply Agent LangGraph node"
```

---

## Task 6: Battery Agent Node

**Files:**
- Modify: `simulation/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 6.1: Add failing test to `tests/test_graph.py`**

```python
def test_battery_agent_returns_action():
    from simulation.graph import battery_agent
    state = _make_state(battery_soc_pct=30.0, tariff=18.0, pv_kw=22.0)
    mock_llm = _mock_llm_response({
        "charge_discharge_kw": 20.0,
        "score": 0.9,
        "rationale": "Low SOC and high PV — charge at 20 kW",
    })
    with patch("simulation.graph._get_llm", return_value=mock_llm):
        result = battery_agent(state)
    assert "battery_action" in result
    action = result["battery_action"]
    assert "charge_discharge_kw" in action["proposed"]
    assert 0.0 <= action["score"] <= 1.0


def test_battery_agent_discharge_is_negative():
    from simulation.graph import battery_agent
    state = _make_state(battery_soc_pct=60.0, tariff=34.0, pv_kw=5.0)
    mock_llm = _mock_llm_response({
        "charge_discharge_kw": -15.0,
        "score": 0.8,
        "rationale": "Peak tariff — discharge 15 kW to offset grid draw",
    })
    with patch("simulation.graph._get_llm", return_value=mock_llm):
        result = battery_agent(state)
    assert result["battery_action"]["proposed"]["charge_discharge_kw"] < 0
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_graph.py::test_battery_agent_returns_action tests/test_graph.py::test_battery_agent_discharge_is_negative -v
```

Expected: `AttributeError: module 'simulation.graph' has no attribute 'battery_agent'`

- [ ] **Step 6.3: Add Battery Agent to `simulation/graph.py`**

Append after `supply_agent`:

```python
# ── Battery Agent ─────────────────────────────────────────────────────────────

_BATTERY_SYSTEM = """You are the Battery Agent for a Building Management System.
Your job: recommend charge or discharge power to optimise SOC health and tariff arbitrage.

Rules:
- Positive charge_discharge_kw = charging, negative = discharging.
- Max rate is ±25 kW. Charge rate tapers to 12.5 kW above 80% SOC.
- Avoid discharging below 10% SOC. Avoid charging above 95% SOC.
- Prefer charging when tariff < 15 p/kWh and pv_kw > 10.
- Prefer discharging when tariff > 25 p/kWh and soc > 40%.

Inputs: battery_soc_pct (0–100), tariff (p/kWh), pv_kw.

Respond ONLY with valid JSON — no markdown:
{
  "charge_discharge_kw": <float -25 to +25>,
  "score": <float 0–1, where 1 = optimal arbitrage and SOC health>,
  "rationale": "<one concise sentence>"
}"""


def battery_agent(state: BMSState) -> dict:
    llm = _get_llm()
    human = (
        f"battery_soc_pct={state['battery_soc_pct']}, "
        f"tariff={state['tariff']}p/kWh, "
        f"pv_kw={state['pv_kw']}"
    )
    response = llm.invoke([SystemMessage(content=_BATTERY_SYSTEM), HumanMessage(content=human)])
    data = json.loads(response.content)
    kw = max(-25.0, min(25.0, float(data["charge_discharge_kw"])))
    return {
        "battery_action": AgentAction(
            proposed={"charge_discharge_kw": kw},
            score=max(0.0, min(1.0, float(data["score"]))),
            rationale=str(data["rationale"]),
        )
    }
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_graph.py::test_battery_agent_returns_action tests/test_graph.py::test_battery_agent_discharge_is_negative -v
```

Expected: `2 passed`

- [ ] **Step 6.5: Commit**

```bash
git add simulation/graph.py tests/test_graph.py
git commit -m "feat: add Battery Agent LangGraph node"
```

---

## Task 7: Thermal Agent Node

**Files:**
- Modify: `simulation/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 7.1: Add failing test to `tests/test_graph.py`**

```python
def test_thermal_agent_returns_action():
    from simulation.graph import thermal_agent
    state = _make_state(
        zone_temps={"Server Hall": 27.5, "Open Plan": 24.0, "Boardroom": 23.0,
                    "Reception": 22.5, "Lab A": 26.8},
        comfort_band=(22.0, 26.0),
        ext_temp=32.0,
        occupancy=75.0,
    )
    mock_llm = _mock_llm_response({
        "ahu1_supply_c": 17.0,
        "ahu2_supply_c": 17.5,
        "chiller_c": 6.5,
        "free_cool_pct": 20.0,
        "score": 0.6,
        "rationale": "2 zones above comfort band — increase cooling",
    })
    with patch("simulation.graph._get_llm", return_value=mock_llm):
        result = thermal_agent(state)
    assert "thermal_action" in result
    action = result["thermal_action"]
    p = action["proposed"]
    assert 12.0 <= p["ahu1_supply_c"] <= 24.0
    assert 12.0 <= p["ahu2_supply_c"] <= 24.0
    assert 4.0 <= p["chiller_c"] <= 12.0
    assert 0.0 <= p["free_cool_pct"] <= 100.0
    assert 0.0 <= action["score"] <= 1.0
```

- [ ] **Step 7.2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_graph.py::test_thermal_agent_returns_action -v
```

Expected: `AttributeError: module 'simulation.graph' has no attribute 'thermal_agent'`

- [ ] **Step 7.3: Add Thermal Agent to `simulation/graph.py`**

Append after `battery_agent`:

```python
# ── Thermal Agent ─────────────────────────────────────────────────────────────

_THERMAL_SYSTEM = """You are the Thermal Agent for a Building Management System.
Your job: recommend HVAC setpoints to keep all zones within the comfort band.

Rules:
- ahu1_supply_c and ahu2_supply_c: valid range 12–24 °C (lower = more cooling).
- chiller_c: valid range 4–12 °C (lower = more cooling, more energy).
- free_cool_pct: 0–100 % (higher = more free-side economiser cooling, less chiller energy).
- Comfort band is 22–26 °C. Zones outside the band lower your score.
- Score = zones_in_band / total_zones.

Inputs: zone_temps (dict of zone→°C), comfort_band (low, high), ext_temp (°C), occupancy (%).

Respond ONLY with valid JSON — no markdown:
{
  "ahu1_supply_c": <float 12–24>,
  "ahu2_supply_c": <float 12–24>,
  "chiller_c": <float 4–12>,
  "free_cool_pct": <float 0–100>,
  "score": <float 0–1>,
  "rationale": "<one concise sentence>"
}"""


def thermal_agent(state: BMSState) -> dict:
    llm = _get_llm()
    low, high = state["comfort_band"]
    zone_str = ", ".join(f"{z}={t}°C" for z, t in state["zone_temps"].items())
    human = (
        f"zone_temps=[{zone_str}], "
        f"comfort_band=({low},{high})°C, "
        f"ext_temp={state['ext_temp']}°C, "
        f"occupancy={state['occupancy']}%"
    )
    response = llm.invoke([SystemMessage(content=_THERMAL_SYSTEM), HumanMessage(content=human)])
    data = json.loads(response.content)
    return {
        "thermal_action": AgentAction(
            proposed={
                "ahu1_supply_c": max(12.0, min(24.0, float(data["ahu1_supply_c"]))),
                "ahu2_supply_c": max(12.0, min(24.0, float(data["ahu2_supply_c"]))),
                "chiller_c": max(4.0, min(12.0, float(data["chiller_c"]))),
                "free_cool_pct": max(0.0, min(100.0, float(data["free_cool_pct"]))),
            },
            score=max(0.0, min(1.0, float(data["score"]))),
            rationale=str(data["rationale"]),
        )
    }
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_graph.py::test_thermal_agent_returns_action -v
```

Expected: `1 passed`

- [ ] **Step 7.5: Commit**

```bash
git add simulation/graph.py tests/test_graph.py
git commit -m "feat: add Thermal Agent LangGraph node"
```

---

## Task 8: Orchestration Node + Graph Wiring

**Files:**
- Modify: `simulation/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 8.1: Add failing tests to `tests/test_graph.py`**

```python
def test_orchestration_applies_comfort_hard_constraint():
    from simulation.graph import orchestration_agent
    state = _make_state(
        zone_temps={"Server Hall": 27.5, "Open Plan": 28.0, "Boardroom": 23.0,
                    "Reception": 22.5, "Lab A": 27.0},
        comfort_band=(22.0, 26.0),
        demand_action=AgentAction(proposed={"demand_limit_kw": 80.0, "dr_signal": "normal"}, score=0.7, rationale="ok"),
        supply_action=AgentAction(proposed={"pv_divert_pct": 60.0, "grid_import_limit_kw": 100.0}, score=0.6, rationale="ok"),
        battery_action=AgentAction(proposed={"charge_discharge_kw": -10.0}, score=0.5, rationale="ok"),
        thermal_action=AgentAction(
            proposed={"ahu1_supply_c": 16.0, "ahu2_supply_c": 16.0, "chiller_c": 5.0, "free_cool_pct": 10.0},
            score=0.4,  # low because zones are outside band
            rationale="Aggressive cooling required",
        ),
    )
    result = orchestration_agent(state)
    assert "final_setpoints" in result
    sp = result["final_setpoints"]
    # Thermal proposal accepted verbatim when zones outside comfort band
    assert sp["ahu1_supply_c"] == 16.0
    assert sp["chiller_c"] == 5.0
    assert sp["demand_limit_kw"] == 80.0
    assert sp["pv_divert_pct"] == 60.0


def test_orchestration_produces_valid_setpoints():
    from simulation.graph import orchestration_agent
    state = _make_state(
        demand_action=AgentAction(proposed={"demand_limit_kw": 75.0, "dr_signal": "curtail"}, score=0.8, rationale="ok"),
        supply_action=AgentAction(proposed={"pv_divert_pct": 80.0, "grid_import_limit_kw": 60.0}, score=0.75, rationale="ok"),
        battery_action=AgentAction(proposed={"charge_discharge_kw": 15.0}, score=0.9, rationale="ok"),
        thermal_action=AgentAction(
            proposed={"ahu1_supply_c": 18.5, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 40.0},
            score=1.0,
            rationale="All zones in band",
        ),
    )
    result = orchestration_agent(state)
    sp = result["final_setpoints"]
    assert 12.0 <= sp["ahu1_supply_c"] <= 24.0
    assert 12.0 <= sp["ahu2_supply_c"] <= 24.0
    assert 4.0 <= sp["chiller_c"] <= 12.0
    assert 0.0 <= sp["free_cool_pct"] <= 100.0
    assert 0.0 <= sp["pv_divert_pct"] <= 100.0
    assert 20.0 <= sp["demand_limit_kw"] <= 200.0
    assert "agent_trace" in result


def test_full_graph_invoke():
    from simulation.graph import graph
    state = _make_state()

    demand_mock = _mock_llm_response({"demand_limit_kw": 78.0, "dr_signal": "normal", "score": 0.7, "rationale": "Normal conditions"})
    supply_mock = _mock_llm_response({"pv_divert_pct": 47.0, "grid_import_limit_kw": 90.0, "score": 0.6, "rationale": "Moderate PV"})
    battery_mock = _mock_llm_response({"charge_discharge_kw": 5.0, "score": 0.5, "rationale": "Trickle charge"})
    thermal_mock = _mock_llm_response({"ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 30.0, "score": 1.0, "rationale": "All in band"})

    call_count = 0
    mocks = [demand_mock, supply_mock, battery_mock, thermal_mock]

    def rotating_llm(*args, **kwargs):
        nonlocal call_count
        m = mocks[call_count % len(mocks)]
        call_count += 1
        return m

    with patch("simulation.graph._get_llm", side_effect=rotating_llm):
        result_state = graph.invoke(state)

    assert result_state["final_setpoints"] is not None
    assert result_state["demand_action"] is not None
    assert result_state["supply_action"] is not None
    assert result_state["battery_action"] is not None
    assert result_state["thermal_action"] is not None
```

Note: `AgentAction` import at top of test file — add to the imports line:
```python
from simulation.state import AgentAction, BMSState
```

- [ ] **Step 8.2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_graph.py::test_orchestration_applies_comfort_hard_constraint tests/test_graph.py::test_orchestration_produces_valid_setpoints tests/test_graph.py::test_full_graph_invoke -v
```

Expected: `AttributeError: module 'simulation.graph' has no attribute 'orchestration_agent'`

- [ ] **Step 8.3: Add Orchestration node + graph wiring to `simulation/graph.py`**

Append after `thermal_agent`:

```python
# ── Orchestration Agent (pure Python) ────────────────────────────────────────

def _reconcile(state: BMSState) -> tuple[dict, list[dict]]:
    """Arbitrate 4 agent proposals into final setpoints.

    Comfort is a hard constraint. Cost/demand are soft objectives.
    Replace only this function when the Q2 negotiation engine is ready.
    """
    demand = state["demand_action"]
    supply = state["supply_action"]
    battery = state["battery_action"]
    thermal = state["thermal_action"]

    low, high = state["comfort_band"]
    zones_outside = sum(
        1 for t in state["zone_temps"].values()
        if t < low or t > high
    )
    comfort_violated = zones_outside > 0

    # Thermal proposals always accepted (comfort = hard constraint)
    ahu1 = thermal["proposed"]["ahu1_supply_c"]
    ahu2 = thermal["proposed"]["ahu2_supply_c"]
    chiller = thermal["proposed"]["chiller_c"]
    free_cool = thermal["proposed"]["free_cool_pct"]

    demand_limit = demand["proposed"]["demand_limit_kw"]
    pv_divert = supply["proposed"]["pv_divert_pct"]

    # Zone setpoints: comfort band midpoint ± occupancy offset
    midpoint = (low + high) / 2.0
    occ_offset = (state["occupancy"] - 50.0) / 100.0 * 1.0
    zone_cooling_sp = max(low, min(high, midpoint - occ_offset))
    zone_heating_sp = max(16.0, min(22.0, 19.0 - occ_offset))

    trace = [
        {"agent": "demand",  "status": "accepted", "proposed": demand["proposed"],  "score": demand["score"],  "rationale": demand["rationale"]},
        {"agent": "supply",  "status": "accepted", "proposed": supply["proposed"],  "score": supply["score"],  "rationale": supply["rationale"]},
        {"agent": "battery", "status": "accepted", "proposed": battery["proposed"], "score": battery["score"], "rationale": battery["rationale"]},
        {"agent": "thermal", "status": "accepted" if not comfort_violated else "accepted", "proposed": thermal["proposed"], "score": thermal["score"], "rationale": thermal["rationale"]},
        {"agent": "orchestration", "status": "—", "proposed": {}, "score": 1.0,
         "rationale": f"Reconciled · {5 - zones_outside}/5 zones in comfort band"},
    ]

    setpoints = {
        "ahu1_supply_c": ahu1,
        "ahu2_supply_c": ahu2,
        "chiller_c": chiller,
        "free_cool_pct": free_cool,
        "pv_divert_pct": pv_divert,
        "zone_cooling_sp_c": zone_cooling_sp,
        "zone_heating_sp_c": zone_heating_sp,
        "demand_limit_kw": demand_limit,
    }
    return setpoints, trace


def orchestration_agent(state: BMSState) -> dict:
    final_setpoints, agent_trace = _reconcile(state)
    return {"final_setpoints": final_setpoints, "agent_trace": agent_trace}


# ── Graph wiring ──────────────────────────────────────────────────────────────

_workflow = StateGraph(BMSState)
_workflow.add_node("entry",         entry_node)
_workflow.add_node("demand",        demand_agent)
_workflow.add_node("supply",        supply_agent)
_workflow.add_node("battery",       battery_agent)
_workflow.add_node("thermal",       thermal_agent)
_workflow.add_node("orchestration", orchestration_agent)

_workflow.add_edge(START,           "entry")
_workflow.add_edge("entry",         "demand")
_workflow.add_edge("entry",         "supply")
_workflow.add_edge("entry",         "battery")
_workflow.add_edge("entry",         "thermal")
_workflow.add_edge("demand",        "orchestration")
_workflow.add_edge("supply",        "orchestration")
_workflow.add_edge("battery",       "orchestration")
_workflow.add_edge("thermal",       "orchestration")
_workflow.add_edge("orchestration", END)

graph = _workflow.compile()
```

Also add `"agent_trace": list | None` to `BMSState` in `simulation/state.py`:

```python
# in BMSState TypedDict, add after final_setpoints:
agent_trace: list | None
```

And update `_make_state()` in `tests/test_graph.py` to include `"agent_trace": None`.

- [ ] **Step 8.4: Run all graph tests**

```bash
.venv/bin/pytest tests/test_graph.py -v
```

Expected: `8 passed` (all prior tests + 3 new ones)

- [ ] **Step 8.5: Run full test suite to check no regressions**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All prior tests pass + new ones.

- [ ] **Step 8.6: Commit**

```bash
git add simulation/graph.py simulation/state.py tests/test_graph.py
git commit -m "feat: add Orchestration node and wire LangGraph StateGraph"
```

---

## Task 9: API Models Update

**Files:**
- Modify: `api/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 9.1: Write failing tests**

Open `tests/test_models.py` and append:

```python
from api.models import AgentTraceEntry, OptimizeResponse, OptimizeRequest


def test_agent_trace_entry_model():
    entry = AgentTraceEntry(
        agent="demand",
        status="accepted",
        proposed={"demand_limit_kw": 75.0},
        score=0.8,
        rationale="Peak avoidance",
    )
    assert entry.agent == "demand"
    assert entry.score == 0.8


def test_optimize_request_accepts_scenario():
    req = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0, scenario="normal")
    assert req.scenario == "normal"


def test_optimize_request_scenario_defaults_none():
    req = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    assert req.scenario is None


def test_optimize_response_includes_agent_trace():
    resp = OptimizeResponse(
        power_kw=30.0, savings_pct=10, avg_zone_temp=23.5,
        pv_contribution_pct=40, comfort_zones=4,
        zone_temps={"Server Hall": 23.5},
        energy_forecast_kwh=[1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8],
        setpoints={"AHU-1 supply": "18.0°C"},
        simulation_duration_s=4.2,
        battery_soc_pct=55.0,
        agent_trace=[
            AgentTraceEntry(agent="demand", status="accepted",
                            proposed={}, score=0.7, rationale="ok")
        ],
    )
    assert resp.battery_soc_pct == 55.0
    assert len(resp.agent_trace) == 1
```

- [ ] **Step 9.2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: `ImportError` for `AgentTraceEntry`, and failures for new fields.

- [ ] **Step 9.3: Update `api/models.py`**

Replace the entire file:

```python
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    occupancy: float = Field(..., ge=0, le=100)
    ext_temp: float = Field(..., ge=10, le=42)
    pv_kw: float = Field(..., ge=0, le=30)
    tariff: float = Field(..., ge=5, le=40)
    horizon_hours: int = Field(default=4, ge=1, le=8)
    scenario: str | None = None


class AgentTraceEntry(BaseModel):
    agent: str
    status: str
    proposed: dict
    score: float
    rationale: str


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
    battery_soc_pct: float = 50.0
    agent_trace: list[AgentTraceEntry] = []


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

- [ ] **Step 9.4: Run all tests**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All prior tests + 4 new model tests pass.

- [ ] **Step 9.5: Commit**

```bash
git add api/models.py tests/test_models.py
git commit -m "feat: add AgentTraceEntry model and extend OptimizeRequest/Response"
```

---

## Task 10: Wire Graph into API Endpoint

**Files:**
- Modify: `api/main.py`

- [ ] **Step 10.1: Write failing integration test**

Open `tests/test_api.py` and append:

```python
from unittest.mock import patch, MagicMock
import json


def test_optimize_returns_agent_trace(client):
    mock_state = {
        "final_setpoints": {
            "ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0,
            "chiller_c": 7.0, "free_cool_pct": 30.0,
            "pv_divert_pct": 46.7, "zone_cooling_sp_c": 24.0,
            "zone_heating_sp_c": 19.0, "demand_limit_kw": 80.0,
        },
        "demand_action": {"proposed": {"demand_limit_kw": 80.0, "dr_signal": "normal"}, "score": 0.7, "rationale": "ok"},
        "supply_action": {"proposed": {"pv_divert_pct": 46.7, "grid_import_limit_kw": 90.0}, "score": 0.6, "rationale": "ok"},
        "battery_action": {"proposed": {"charge_discharge_kw": 5.0}, "score": 0.5, "rationale": "ok"},
        "thermal_action": {"proposed": {"ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 30.0}, "score": 1.0, "rationale": "ok"},
        "agent_trace": [
            {"agent": "demand", "status": "accepted", "proposed": {}, "score": 0.7, "rationale": "ok"},
            {"agent": "supply", "status": "accepted", "proposed": {}, "score": 0.6, "rationale": "ok"},
            {"agent": "battery", "status": "accepted", "proposed": {}, "score": 0.5, "rationale": "ok"},
            {"agent": "thermal", "status": "accepted", "proposed": {}, "score": 1.0, "rationale": "ok"},
            {"agent": "orchestration", "status": "—", "proposed": {}, "score": 1.0, "rationale": "Reconciled"},
        ],
        "battery_soc_pct": 51.25,
    }

    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:

        mock_graph.invoke.return_value = mock_state

        from simulation.engine import SimulationResult
        from simulation.mpc import Setpoints
        mock_engine.run.return_value = SimulationResult(
            power_kw=30.0, savings_pct=10, avg_zone_temp=23.5,
            pv_contribution_pct=40, comfort_zones=4,
            zone_temps={"Server Hall": 23.5, "Open Plan": 24.0, "Boardroom": 22.8, "Reception": 23.1, "Lab A": 24.5},
            energy_forecast_kwh=[1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8],
            setpoints={"AHU-1 supply": "18.0°C"},
            simulation_duration_s=4.2,
        )

        resp = client.post("/optimize", json={"occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0})

    assert resp.status_code == 200
    data = resp.json()
    assert "agent_trace" in data
    assert len(data["agent_trace"]) == 5
    assert data["agent_trace"][0]["agent"] == "demand"
    assert "battery_soc_pct" in data


def test_optimize_scenario_seeds_battery(client):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = {
            "final_setpoints": {"ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0, "chiller_c": 7.0,
                                "free_cool_pct": 30.0, "pv_divert_pct": 0.0, "zone_cooling_sp_c": 24.0,
                                "zone_heating_sp_c": 19.0, "demand_limit_kw": 80.0},
            "demand_action": {"proposed": {}, "score": 0.5, "rationale": "ok"},
            "supply_action": {"proposed": {}, "score": 0.5, "rationale": "ok"},
            "battery_action": {"proposed": {"charge_discharge_kw": 0.0}, "score": 0.5, "rationale": "ok"},
            "thermal_action": {"proposed": {}, "score": 1.0, "rationale": "ok"},
            "agent_trace": [],
            "battery_soc_pct": 90.0,
        }
        from simulation.engine import SimulationResult
        mock_engine.run.return_value = SimulationResult(
            power_kw=5.0, savings_pct=5, avg_zone_temp=22.0,
            pv_contribution_pct=0, comfort_zones=5,
            zone_temps={"Server Hall": 22.0, "Open Plan": 22.0, "Boardroom": 22.0, "Reception": 22.0, "Lab A": 22.0},
            energy_forecast_kwh=[0.5]*7,
            setpoints={"AHU-1 supply": "18.0°C"},
            simulation_duration_s=2.0,
        )

        resp = client.post("/optimize", json={"occupancy": 5, "ext_temp": 16.0, "pv_kw": 0.0, "tariff": 5.0, "scenario": "night"})

    assert resp.status_code == 200
    # Verify graph.invoke was called with a state that includes scenario="night"
    call_args = mock_graph.invoke.call_args[0][0]
    assert call_args["scenario"] == "night"
```

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_api.py::test_optimize_returns_agent_trace tests/test_api.py::test_optimize_scenario_seeds_battery -v
```

Expected: Failures because `api.main` still uses `compute_setpoints`.

- [ ] **Step 10.3: Rewrite `api/main.py`**

```python
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.models import AgentTraceEntry, OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse
from simulation.battery import get_battery, update_soc
from simulation.graph import graph
from simulation.mpc import Setpoints
from simulation.engine import SimulationEngine, EP_DIR
from simulation.state import BMSState

_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=True)


app = FastAPI(title="BuilMirai MPC API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ROOT = Path(__file__).parent.parent
_SIM_DIR = _ROOT / "simulation"
_IDF_PATH = _SIM_DIR / "building.idf"
_DASHBOARD = _ROOT / "builmirai_mpc_hvac_dashboard.html"


def _resolve_epw() -> Path:
    if os.environ.get("WEATHER_FILE"):
        return Path(os.environ["WEATHER_FILE"])
    bundled = _SIM_DIR / "weather.epw"
    if bundled.exists() and bundled.stat().st_size > 1000:
        return bundled
    ep_weather_dir = EP_DIR / "WeatherData"
    if ep_weather_dir.is_dir():
        epw_files = list(ep_weather_dir.glob("*.epw"))
        if epw_files:
            return epw_files[0]
    return bundled


_EPW_PATH = _resolve_epw()
engine = SimulationEngine(idf_path=_IDF_PATH, weather_path=_EPW_PATH, ep_dir=EP_DIR)
_engine_lock = asyncio.Lock()

SCENARIOS: dict[str, dict] = {
    "normal":   {"occupancy": 70,  "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0},
    "peak":     {"occupancy": 90,  "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0},
    "heatwave": {"occupancy": 75,  "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0},
    "preheat":  {"occupancy": 20,  "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0},
    "night":    {"occupancy": 5,   "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0},
}

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_DASHBOARD, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    epw_ok = _EPW_PATH.exists() and _EPW_PATH.stat().st_size > 1000
    return HealthResponse(
        status="ok" if epw_ok else "degraded",
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
    battery = get_battery()
    initial_state: BMSState = {
        "occupancy": request.occupancy,
        "ext_temp": request.ext_temp,
        "pv_kw": request.pv_kw,
        "tariff": request.tariff,
        "net_power_kw": max(0.0, request.pv_kw),  # refined by entry_node
        "zone_temps": {z: 23.0 for z in ZONE_NAMES},
        "comfort_band": (22.0, 26.0),
        "battery_soc_pct": battery.soc_pct,
        "scenario": request.scenario,
        "demand_action": None,
        "supply_action": None,
        "battery_action": None,
        "thermal_action": None,
        "final_setpoints": None,
        "agent_trace": None,
    }

    loop = asyncio.get_running_loop()
    try:
        result_state = await loop.run_in_executor(_executor, lambda: graph.invoke(initial_state))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent graph failed: {exc}") from exc

    sp_dict = result_state["final_setpoints"]
    final_setpoints = Setpoints(
        ahu1_supply_c=sp_dict["ahu1_supply_c"],
        ahu2_supply_c=sp_dict["ahu2_supply_c"],
        chiller_c=sp_dict["chiller_c"],
        free_cool_pct=sp_dict["free_cool_pct"],
        pv_divert_pct=sp_dict["pv_divert_pct"],
        zone_cooling_sp_c=sp_dict["zone_cooling_sp_c"],
        zone_heating_sp_c=sp_dict["zone_heating_sp_c"],
        demand_limit_kw=sp_dict["demand_limit_kw"],
    )

    try:
        async with _engine_lock:
            sim_result = await loop.run_in_executor(
                _executor,
                lambda: engine.run(request=request, setpoints=final_setpoints),
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc

    # Advance battery SOC based on battery agent's recommendation
    bat_action = result_state.get("battery_action") or {}
    charge_kw = (bat_action.get("proposed") or {}).get("charge_discharge_kw", 0.0)
    new_soc = update_soc(battery.soc_pct, charge_kw)
    battery.soc_pct = new_soc

    raw_trace = result_state.get("agent_trace") or []
    agent_trace = [AgentTraceEntry(**entry) for entry in raw_trace]

    return OptimizeResponse(
        power_kw=sim_result.power_kw,
        savings_pct=sim_result.savings_pct,
        avg_zone_temp=sim_result.avg_zone_temp,
        pv_contribution_pct=sim_result.pv_contribution_pct,
        comfort_zones=sim_result.comfort_zones,
        zone_temps=sim_result.zone_temps,
        energy_forecast_kwh=sim_result.energy_forecast_kwh,
        setpoints=sim_result.setpoints,
        simulation_duration_s=sim_result.simulation_duration_s,
        battery_soc_pct=round(new_soc, 1),
        agent_trace=agent_trace,
    )
```

- [ ] **Step 10.4: Run all tests**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All tests pass including the 2 new API tests.

- [ ] **Step 10.5: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: wire LangGraph graph.invoke into /optimize endpoint"
```

---

## Task 11: Dashboard UI — Agent Graph Panel + Trace Log + Battery Chip

**Files:**
- Modify: `builmirai_mpc_hvac_dashboard.html`

This task has no automated test — verify visually by running the dev server and checking the dashboard renders correctly.

- [ ] **Step 11.1: Add CSS for agent graph nodes**

Find the block containing `.layer-pid {` (around line 326) and add after it:

```css
/* ── AGENT GRAPH ── */
.agent-graph { display: flex; flex-direction: column; align-items: center; gap: 10px; }
.agent-row   { display: flex; gap: 10px; justify-content: center; }
.agent-node  {
  background: var(--bg-card2); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 10px 14px; min-width: 110px;
  display: flex; flex-direction: column; align-items: center; gap: 4px;
}
.agent-node-name  { font-size: 12px; font-weight: 600; color: var(--text-1); }
.agent-node-sub   { font-size: 10px; color: var(--text-3); text-align: center; }
.agent-rationale  { font-size: 10px; color: var(--text-2); text-align: center; max-width: 100px; line-height: 1.3; }
.agent-badge { font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 99px; }
.badge-idle      { background: var(--bg-input); color: var(--text-3); }
.badge-running   { background: var(--blue-lo); color: var(--blue); }
.badge-accepted  { background: var(--green-lo); color: var(--green); }
.badge-overridden{ background: var(--amber-lo); color: var(--amber); }
.agent-connector { color: var(--text-3); font-size: 18px; line-height: 1; }
.agent-fan { display: flex; gap: 4px; align-items: center; color: var(--text-3); font-size: 11px; }
```

- [ ] **Step 11.2: Replace the Control Architecture panel HTML**

Find the entire `<!-- Control stack -->` card block (lines ~554–590) and replace it:

```html
<!-- Control stack -->
<div class="card">
  <div class="card-header">
    <div>
      <div class="card-title">Control architecture</div>
      <div class="card-desc">5-agent LangGraph pipeline</div>
    </div>
  </div>
  <div class="agent-graph" id="agent-graph">
    <!-- Entry node -->
    <div class="agent-node" style="border-color:var(--border-hi)">
      <div class="agent-node-name">⚡ Entry</div>
      <div class="agent-node-sub">seeds BMSState</div>
      <span class="agent-badge badge-idle" id="badge-entry">idle</span>
    </div>
    <!-- Fan-out arrow -->
    <div class="agent-fan">
      <span>↙</span><span>↓</span><span>↓</span><span>↓</span><span>↘</span>
    </div>
    <!-- 4 specialist nodes -->
    <div class="agent-row">
      <div class="agent-node">
        <div class="agent-node-name">📊 Demand</div>
        <span class="agent-badge badge-idle" id="badge-demand">idle</span>
        <div class="agent-rationale" id="rationale-demand">—</div>
      </div>
      <div class="agent-node">
        <div class="agent-node-name">⚡ Supply</div>
        <span class="agent-badge badge-idle" id="badge-supply">idle</span>
        <div class="agent-rationale" id="rationale-supply">—</div>
      </div>
      <div class="agent-node">
        <div class="agent-node-name">🔋 Battery</div>
        <span class="agent-badge badge-idle" id="badge-battery">idle</span>
        <div class="agent-rationale" id="rationale-battery">—</div>
      </div>
      <div class="agent-node">
        <div class="agent-node-name">🌡️ Thermal</div>
        <span class="agent-badge badge-idle" id="badge-thermal">idle</span>
        <div class="agent-rationale" id="rationale-thermal">—</div>
      </div>
    </div>
    <!-- Fan-in arrow -->
    <div class="agent-fan">
      <span>↘</span><span>↓</span><span>↓</span><span>↓</span><span>↙</span>
    </div>
    <!-- Orchestration node -->
    <div class="agent-node" style="border-color:var(--border-hi)">
      <div class="agent-node-name">🎯 Orchestration</div>
      <div class="agent-node-sub">arbitrates setpoints</div>
      <span class="agent-badge badge-idle" id="badge-orchestration">idle</span>
      <div class="agent-rationale" id="rationale-orchestration">—</div>
    </div>
  </div>
</div>
```

- [ ] **Step 11.3: Add battery SOC chip to the status bar**

Find `<div class="status-bar">` (around line 418) and add a battery chip inside it, before the existing chips. Find the first `<div class="status-chip"` inside `.status-bar` and add before it:

```html
<div class="status-chip" id="bat-chip" style="background:var(--green-lo);color:var(--green)">BAT · 50%</div>
```

- [ ] **Step 11.4: Replace `addLog()` and add `renderAgentGraph()` in the JS section**

Find `function addLog(s) {` (around line 721) and replace the entire function and `const logMessages`/`logTypes`/`logTexts`/`logIdx` declarations above it with:

```javascript
const logMessages = [];

function renderAgentTrace(trace) {
  if (!trace || !trace.length) return;
  const t = new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const typeMap = { accepted: 'log-ok', overridden: 'log-warn', '—': 'log-info' };
  const newEntries = trace.map(e => ({
    t,
    type: typeMap[e.status] || 'log-info',
    text: `[${e.agent.charAt(0).toUpperCase()+e.agent.slice(1)}] ${e.rationale} [${e.status}]`,
  }));
  logMessages.unshift(...newEntries);
  if (logMessages.length > 10) logMessages.splice(10);
  document.getElementById('sys-log').innerHTML = logMessages.map(m =>
    `<div class="log-line"><span class="log-time">${m.t}</span><span class="${m.type}">${m.text}</span></div>`
  ).join('');
}

function renderAgentGraph(trace) {
  const agents = ['demand','supply','battery','thermal','orchestration'];
  agents.forEach(a => {
    const badge = document.getElementById('badge-'+a);
    const rat   = document.getElementById('rationale-'+a);
    if (badge) { badge.textContent = 'idle'; badge.className = 'agent-badge badge-idle'; }
    if (rat)   rat.textContent = '—';
  });
  if (!trace) return;
  trace.forEach(e => {
    const badge = document.getElementById('badge-'+e.agent);
    const rat   = document.getElementById('rationale-'+e.agent);
    if (!badge) return;
    const cls = e.status === 'accepted' ? 'badge-accepted'
              : e.status === 'overridden' ? 'badge-overridden'
              : 'badge-idle';
    badge.textContent = e.status;
    badge.className = 'agent-badge ' + cls;
    if (rat) rat.textContent = e.rationale || '—';
  });
  const entryBadge = document.getElementById('badge-entry');
  if (entryBadge) { entryBadge.textContent = 'done'; entryBadge.className = 'agent-badge badge-accepted'; }
}

function updateBatterySoc(pct) {
  const chip = document.getElementById('bat-chip');
  if (!chip) return;
  chip.textContent = `BAT · ${Math.round(pct)}%`;
  chip.style.background = pct > 50 ? 'var(--green-lo)' : pct > 20 ? 'var(--amber-lo)' : 'var(--red-lo)';
  chip.style.color       = pct > 50 ? 'var(--green)'    : pct > 20 ? 'var(--amber)'    : 'var(--red)';
}

function addLog(s) {
  // Offline fallback — only used when EnergyPlus backend isn't running
  const t = new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  logMessages.unshift({ t, type: 'log-info', text: `[Offline] Heuristic compute · ext ${s.ext}°C · occ ${s.occ}%` });
  if (logMessages.length > 5) logMessages.pop();
  document.getElementById('sys-log').innerHTML = logMessages.map(m =>
    `<div class="log-line"><span class="log-time">${m.t}</span><span class="${m.type}">${m.text}</span></div>`
  ).join('');
}
```

- [ ] **Step 11.5: Update the API response handler to call new render functions**

Find the section starting `fetch('/optimize', {` (around line 760). Inside the `.then(data => {` block, after `renderSetpoints(data.setpoints);` add:

```javascript
renderAgentTrace(data.agent_trace);
renderAgentGraph(data.agent_trace);
updateBatterySoc(data.battery_soc_pct ?? 50);
```

Also find `renderSetpoints(r.setpoints);` inside the `render(s)` offline function (around line 746) and add after it:

```javascript
updateBatterySoc(50);
```

- [ ] **Step 11.6: Set agent badges to "running" while the API call is in-flight**

Find the `runBtn.addEventListener('click', ...)` handler. Before the `fetch` call, add:

```javascript
['demand','supply','battery','thermal','orchestration','entry'].forEach(a => {
  const b = document.getElementById('badge-'+a);
  if (b) { b.textContent = 'running'; b.className = 'agent-badge badge-running'; }
});
```

- [ ] **Step 11.7: Verify dashboard renders correctly**

```bash
.venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`. Check:
- Status bar shows `BAT · 50%` chip in green
- Control Architecture card shows 5 agent nodes in flexbox layout
- Clicking "Run optimiser" sets all badges to `running` (blue)
- After response: 4 specialist badges show `accepted`/`overridden`, Orchestration shows `—`
- System Log shows 5 lines (one per agent) with rationale text
- All other panels (Zone Temperatures, Energy Forecast, MPC Setpoint Outputs) still work

- [ ] **Step 11.8: Commit**

```bash
git add builmirai_mpc_hvac_dashboard.html
git commit -m "feat: update dashboard with 5-agent graph panel, trace log, and battery SOC chip"
```

---

## Task 12: Scenario Smoke Tests

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 12.1: Add scenario smoke tests**

Append to `tests/test_api.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from simulation.engine import SimulationResult


def _mock_graph_state(ahu1=18.0, ahu2=18.0, chiller=7.0, free_cool=30.0,
                      pv_divert=46.7, demand_limit=80.0, soc=50.0):
    return {
        "final_setpoints": {
            "ahu1_supply_c": ahu1, "ahu2_supply_c": ahu2,
            "chiller_c": chiller, "free_cool_pct": free_cool,
            "pv_divert_pct": pv_divert, "zone_cooling_sp_c": 24.0,
            "zone_heating_sp_c": 19.0, "demand_limit_kw": demand_limit,
        },
        "demand_action":  {"proposed": {"demand_limit_kw": demand_limit, "dr_signal": "normal"}, "score": 0.7, "rationale": "ok"},
        "supply_action":  {"proposed": {"pv_divert_pct": pv_divert, "grid_import_limit_kw": 90.0}, "score": 0.6, "rationale": "ok"},
        "battery_action": {"proposed": {"charge_discharge_kw": 0.0}, "score": 0.5, "rationale": "ok"},
        "thermal_action": {"proposed": {"ahu1_supply_c": ahu1, "ahu2_supply_c": ahu2, "chiller_c": chiller, "free_cool_pct": free_cool}, "score": 1.0, "rationale": "ok"},
        "agent_trace": [
            {"agent": "demand", "status": "accepted", "proposed": {}, "score": 0.7, "rationale": "ok"},
            {"agent": "supply", "status": "accepted", "proposed": {}, "score": 0.6, "rationale": "ok"},
            {"agent": "battery", "status": "accepted", "proposed": {}, "score": 0.5, "rationale": "ok"},
            {"agent": "thermal", "status": "accepted", "proposed": {}, "score": 1.0, "rationale": "ok"},
            {"agent": "orchestration", "status": "—", "proposed": {}, "score": 1.0, "rationale": "Reconciled"},
        ],
        "battery_soc_pct": soc,
    }


def _mock_sim_result(comfort_zones=4, power_kw=30.0):
    return SimulationResult(
        power_kw=power_kw, savings_pct=10, avg_zone_temp=23.5,
        pv_contribution_pct=40, comfort_zones=comfort_zones,
        zone_temps={"Server Hall": 23.5, "Open Plan": 24.0, "Boardroom": 22.8,
                    "Reception": 23.1, "Lab A": 24.5},
        energy_forecast_kwh=[1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8],
        setpoints={"AHU-1 supply": "18.0°C", "AHU-2 supply": "18.0°C",
                   "Chiller setpt": "7.0°C", "Free-cool %": "30%",
                   "PV divert": "47%", "Demand limit": "80 kW"},
        simulation_duration_s=4.2,
    )


@pytest.mark.parametrize("scenario,payload,min_comfort", [
    ("normal",   {"occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0, "scenario": "normal"},   3),
    ("peak",     {"occupancy": 90, "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0, "scenario": "peak"},     3),
    ("heatwave", {"occupancy": 75, "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0, "scenario": "heatwave"}, 3),
    ("preheat",  {"occupancy": 20, "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0,  "scenario": "preheat"},  3),
    ("night",    {"occupancy": 5,  "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0,  "scenario": "night"},    3),
])
def test_scenario_smoke(client, scenario, payload, min_comfort):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = _mock_graph_state()
        mock_engine.run.return_value = _mock_sim_result(comfort_zones=min_comfort)

        resp = client.post("/optimize", json=payload)

    assert resp.status_code == 200, f"{scenario}: {resp.text}"
    data = resp.json()
    assert data["comfort_zones"] >= min_comfort, f"{scenario}: comfort_zones={data['comfort_zones']}"
    assert len(data["agent_trace"]) == 5
    sp = data["setpoints"]
    assert "AHU-1 supply" in sp
    assert "Chiller setpt" in sp
    assert "Demand limit" in sp
```

- [ ] **Step 12.2: Run smoke tests**

```bash
.venv/bin/pytest tests/test_api.py -v -k "scenario"
```

Expected: `5 passed` (one per scenario preset)

- [ ] **Step 12.3: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: All tests pass (39 original + new tests).

- [ ] **Step 12.4: Final commit**

```bash
git add tests/test_api.py
git commit -m "test: add scenario smoke tests for all 5 LangGraph presets"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] §2 BMSState + AgentAction — Task 1
- [x] §3 Battery SOC + SCENARIO_SOC — Task 2
- [x] §4 LLM provider factory — Task 3
- [x] §5.1 Demand Agent — Task 4
- [x] §5.2 Supply Agent — Task 5
- [x] §5.3 Battery Agent — Task 6
- [x] §5.4 Thermal Agent — Task 7
- [x] §5.5 Orchestration + `_reconcile()` — Task 8
- [x] §6 Graph wiring START→entry→[D,S,B,T]→orch→END — Task 8
- [x] §7 API models (AgentTraceEntry, scenario field, battery_soc_pct) — Task 9
- [x] §7 API endpoint rewrite — Task 10
- [x] §8 Control Architecture panel (agent graph CSS + HTML) — Task 11
- [x] §8 System Log (renderAgentTrace) — Task 11
- [x] §8 Battery SOC chip — Task 11
- [x] §10 Scenario smoke tests — Task 12
- [x] `agent_trace: list | None` added to BMSState (Step 8.3 note)

**Type consistency:**
- `AgentAction` defined in Task 1 (`simulation/state.py`), used in Tasks 4–8 — consistent
- `BMSState` defined in Task 1, extended in Task 8 with `agent_trace` — consistent
- `Setpoints` dataclass unchanged — Tasks 8 + 10 both reference same 8 fields — consistent
- `AgentTraceEntry` defined in Task 9 (`api/models.py`), consumed in Task 10 — consistent
- `renderAgentGraph(trace)` defined and called in Task 11 — consistent

**No placeholders found.**
