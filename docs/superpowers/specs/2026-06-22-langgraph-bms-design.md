# BuilMirai LangGraph 5-Agent Architecture — Design Spec

**Date:** 2026-06-22
**Status:** Approved
**Scope:** Refactor single-heuristic MPC block into a 5-agent LangGraph StateGraph while keeping EnergyPlus simulation, UI panels, and live telemetry working.

---

## 1. Context and Goals

The current system has a single `compute_setpoints()` function in `simulation/mpc.py` that applies 5 additive heuristic rules to produce HVAC setpoints. The goal is to replace this with a proper multi-agent architecture using LangGraph, where 4 LLM-backed specialist agents reason about their domain in parallel, and a 5th Orchestration agent reconciles their proposals into final setpoints.

**What stays unchanged:**
- `simulation/engine.py` — EnergyPlus integration
- `simulation/mpc.py` — `Setpoints` dataclass (still used by engine)
- All existing API response fields consumed by the dashboard
- `renderSetpoints()`, `renderZones()`, `renderForecast()` JS functions

**What changes:**
- `compute_setpoints()` is replaced by `graph.invoke(state)`
- Battery SOC simulation is added (new component)
- Control Architecture panel updated from 3-layer to 5-agent graph view
- System Log updated to surface per-agent LLM rationale
- Battery SOC chip added to status bar

---

## 2. Shared State Schema

**File:** `simulation/state.py`

```python
from typing import TypedDict

class AgentAction(TypedDict):
    proposed: dict      # agent-specific setpoint proposal
    score: float        # objective score 0–1 (used by Orchestration for weighting)
    rationale: str      # one-line explanation surfaced in System Log

class BMSState(TypedDict):
    # ── Inputs (populated by entry_node, read-only for specialist agents) ──
    occupancy: float              # 0–100 %
    ext_temp: float               # °C
    pv_kw: float                  # kW solar generation
    tariff: float                 # p/kWh grid tariff
    net_power_kw: float           # gross demand − PV (derived in entry_node)
    zone_temps: dict[str, float]  # last known zone temps per zone name
    comfort_band: tuple[float, float]  # (22.0, 26.0) °C — hard constraint
    battery_soc_pct: float        # 0–100 %, scenario-seeded each cycle
    scenario: str | None          # active scenario name for SOC seeding

    # ── Agent outputs (each agent writes only its own key) ──
    demand_action: AgentAction | None
    supply_action: AgentAction | None
    battery_action: AgentAction | None
    thermal_action: AgentAction | None

    # ── Orchestration output ──
    final_setpoints: dict | None  # matches Setpoints dataclass fields
```

All agent state keys default to `None` at graph entry. LangGraph's default reducer merges partial dicts returned by each node without conflict since each node writes a distinct key.

---

## 3. Battery SOC Simulation

**File:** `simulation/battery.py`

```python
@dataclass
class BatteryState:
    capacity_kwh: float = 100.0
    max_rate_kw:  float = 25.0
    soc_pct:      float = 50.0
```

**Scenario-seeded starting SOC:**

| Scenario | Starting SOC | Rationale |
|----------|-------------|-----------|
| normal   | 50%         | Mid-state, balanced decisions |
| peak     | 60%         | Some charge available to discharge during peak tariff |
| heatwave | 30%         | Depleted — Battery Agent must prioritise charging from PV |
| preheat  | 80%         | Nearly full — available to assist pre-cooling |
| night    | 90%         | Overnight charging complete |

**`update_soc(current_pct, charge_discharge_kw, dt_minutes=15) -> float`:**
- Positive `charge_discharge_kw` = charging, negative = discharging
- Clamps result to [0, 100] %
- Degradation proxy: max charge rate tapers to 12.5 kW (C/2) above 80% SOC
- Module-level singleton `_battery` persists SOC across solve cycles within a session

---

## 4. LLM Provider Abstraction

**File:** `simulation/llm_provider.py`

```python
def _get_llm(temperature: float = 0.1) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq")
    model    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    if provider == "groq":
        return ChatGroq(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
```

Swapping provider requires only env var changes — zero code changes. Future providers (Anthropic, OpenAI, Ollama) added here without touching agent nodes.

**Required env vars:**
- `GROQ_API_KEY` — Groq API key
- `LLM_PROVIDER` — default `groq`
- `LLM_MODEL` — default `llama-3.3-70b-versatile`

---

## 5. Agent Node Design

**File:** `simulation/graph.py`

All 4 specialist nodes share the same structure:
1. Extract relevant state slice
2. Build structured prompt (system + human message)
3. Call `_get_llm()` chain with `json_mode` output
4. Parse response into `AgentAction`
5. Return `{"{domain}_action": AgentAction}`

### 5.1 Demand Agent

**Reads:** `net_power_kw`, `occupancy`, `tariff`
**Proposes:**
```json
{
  "demand_limit_kw": 75.0,
  "dr_signal": "curtail" | "normal" | "flex"
}
```
**System prompt scope:** Load forecasting, demand response, peak avoidance. Score reflects how much headroom the proposal creates vs. the demand limit.

### 5.2 Supply Agent

**Reads:** `tariff`, `pv_kw`, `battery_soc_pct`
**Proposes:**
```json
{
  "pv_divert_pct": 80.0,
  "grid_import_limit_kw": 50.0
}
```
**System prompt scope:** Grid/PV arbitrage, import/export constraints, tariff optimisation. Score reflects cost savings vs. baseline.

### 5.3 Battery Agent

**Reads:** `battery_soc_pct`, `tariff`, `pv_kw`
**Proposes:**
```json
{
  "charge_discharge_kw": -15.0
}
```
Positive = charge, negative = discharge.
**System prompt scope:** SOC management, degradation avoidance (taper above 80% SOC), tariff-arbitrage discharge. Score reflects SOC health and arbitrage value.

### 5.4 Thermal Agent

**Reads:** `zone_temps`, `comfort_band`, `ext_temp`, `occupancy`
**Proposes:**
```json
{
  "ahu1_supply_c": 18.5,
  "ahu2_supply_c": 18.0,
  "chiller_c": 7.0,
  "free_cool_pct": 45.0
}
```
**System prompt scope:** Zone comfort maintenance, HVAC efficiency, free-cooling opportunity. Score reflects comfort compliance (zones within band / total zones).

### 5.5 Orchestration Agent (pure Python, no LLM)

**Reads:** All 4 `AgentAction` dicts from state.
**Arbitration rules (in priority order):**
1. **Comfort hard constraint:** If any zone is outside `comfort_band`, Thermal Agent's AHU/chiller proposals are accepted verbatim regardless of score.
2. **Demand hard constraint:** `demand_limit_kw` from Demand Agent is always applied as a cap.
3. **Supply + Battery soft objectives:** `pv_divert_pct` weighted by Supply Agent score; `charge_discharge_kw` applied if battery SOC allows.
4. **Remaining setpoints:** Filled from Thermal Agent proposal (already accepted above).
5. **Zone setpoints:** Derived from comfort band midpoint ± occupancy offset.

Writes `final_setpoints` dict with all `Setpoints` dataclass fields. Also writes `agent_trace` list (one `AgentTraceEntry` per agent, marking each as `accepted` or `overridden`).

The arbitration logic is isolated in a single `_reconcile(state)` function — the Q2 negotiation engine replaces only this function without touching the graph structure or any other node.

---

## 6. Graph Wiring

```
START → entry_node → [demand_agent, supply_agent, battery_agent, thermal_agent]
      → orchestration_agent → END
```

```python
workflow = StateGraph(BMSState)
workflow.add_node("entry",         entry_node)
workflow.add_node("demand",        demand_agent)
workflow.add_node("supply",        supply_agent)
workflow.add_node("battery",       battery_agent)
workflow.add_node("thermal",       thermal_agent)
workflow.add_node("orchestration", orchestration_agent)

workflow.add_edge(START,           "entry")
workflow.add_edge("entry",         "demand")
workflow.add_edge("entry",         "supply")
workflow.add_edge("entry",         "battery")
workflow.add_edge("entry",         "thermal")
workflow.add_edge("demand",        "orchestration")
workflow.add_edge("supply",        "orchestration")
workflow.add_edge("battery",       "orchestration")
workflow.add_edge("thermal",       "orchestration")
workflow.add_edge("orchestration", END)

graph = workflow.compile()
```

Graph compiled once at module load. Each `/optimize` call invokes `graph.invoke(initial_state)`.

---

## 7. API Changes

### `api/models.py`

**New model:**
```python
class AgentTraceEntry(BaseModel):
    agent:     str    # demand | supply | battery | thermal | orchestration
    status:    str    # accepted | overridden | partial
    proposed:  dict
    score:     float
    rationale: str
```

**`OptimizeRequest` addition:**
```python
scenario: str | None = None   # seeds battery SOC from SCENARIO_SOC map
```

**`OptimizeResponse` addition:**
```python
agent_trace:    list[AgentTraceEntry]
battery_soc_pct: float
```

All existing fields unchanged.

### `api/main.py`

`/optimize` endpoint flow:
1. Build `BMSState` from `OptimizeRequest` + `battery.get_soc()`
2. `result_state = graph.invoke(initial_state)`
3. `final_setpoints = Setpoints(**result_state["final_setpoints"])`
4. `sim_result = engine.run(request, final_setpoints)` — unchanged
5. `battery.update_soc(result_state["battery_action"]["proposed"]["charge_discharge_kw"])`
6. Return `OptimizeResponse` with `agent_trace` and `battery_soc_pct`

---

## 8. UI Changes

### Control Architecture Panel

Replaces the 3-layer vertical stack. New layout:

```
[ Entry Node ]
      |
[ Demand ] [ Supply ] [ Battery ] [ Thermal ]   ← flexbox row
      |         |          |           |
      └─────────┴──────────┴───────────┘
                      |
             [ Orchestration ]
```

Each agent node card shows:
- Agent name + domain icon
- Status badge: `idle` (--text-3) | `running` (--blue) | `accepted` (--green) | `overridden` (--amber)
- One-line rationale from `agent_trace[i].rationale`

Updated by `renderAgentGraph(trace)` called from the API response handler.

### System Log

`addLog()` replaced by `renderAgentTrace(trace)`. Produces one log line per agent:
```
[HH:MM:SS] [Demand]       → Curtail demand to 72 kW · peak avoidance  [accepted]
[HH:MM:SS] [Supply]       → Divert 80% PV · tariff 34p               [accepted]
[HH:MM:SS] [Battery]      → Discharge 15 kW · SOC 60%→52%            [accepted]
[HH:MM:SS] [Thermal]      → AHU-1 18.5°C · 3/5 zones in band         [accepted]
[HH:MM:SS] [Orchestration]→ Final setpoints reconciled · comfort OK   [—]
```

### Battery SOC Chip

New chip in the existing status bar between PV metric and the live dot:
```html
<div class="status-chip" id="bat-chip">BAT · 60%</div>
```
Colour: green >50%, amber 20–50%, red <20%. Updated from `battery_soc_pct` in API response.

---

## 9. File Map

| File | Change |
|------|--------|
| `simulation/state.py` | New — `BMSState`, `AgentAction` TypedDicts |
| `simulation/battery.py` | New — `BatteryState`, scenario SOC seeds, `update_soc()` |
| `simulation/llm_provider.py` | New — `_get_llm()` factory |
| `simulation/graph.py` | New — 5 node functions + compiled `StateGraph` |
| `simulation/mpc.py` | Unchanged (`Setpoints` dataclass still used by engine) |
| `simulation/engine.py` | Unchanged |
| `api/models.py` | +`AgentTraceEntry`, +`agent_trace`, +`battery_soc_pct`, +`scenario` |
| `api/main.py` | `graph.invoke()` replaces `compute_setpoints()` |
| `builmirai_mpc_hvac_dashboard.html` | Agent graph panel, trace-based log, battery chip |

---

## 10. Testing

- Existing `tests/test_mpc.py` — kept, still tests `Setpoints` bounds
- New `tests/test_graph.py` — mocks Groq calls, asserts `BMSState` transitions and Orchestration arbitration rules
- New `tests/test_battery.py` — unit tests for `update_soc()` including SOC clamping and degradation taper
- Scenario smoke test: run all 5 presets, assert `comfort_zones >= 3` and setpoints within physical bounds

---

## 11. Out of Scope

- Making `engine.run()` async (deferred to a future async refactor)
- Real degradation model for battery (SOC taper proxy is sufficient for demo)
- Q2 negotiation engine (Orchestration `_reconcile()` is a placeholder)
- Authentication / rate limiting for Groq API calls
