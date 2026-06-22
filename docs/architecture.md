# BuilMirai Agentic BMS — Architecture Diagram

```mermaid
flowchart TB
    %% ── Browser ──────────────────────────────────────────────────────
    subgraph Browser["🌐  Browser  (builmirai_mpc_hvac_dashboard.html)"]
        direction LR
        UI_IN["Sliders\nOccupancy · Ext Temp\nSolar PV · Grid Tariff"]
        UI_OUT["Metrics Panel\nZone Temps · Forecast\nAgent Trace · BAT chip"]
    end

    %% ── FastAPI ───────────────────────────────────────────────────────
    subgraph API["🚀  FastAPI Server  (api/main.py)"]
        direction TB
        OPT["POST /optimize"]
        HEALTH["GET /health\nGET /scenarios/{name}"]
    end

    %% ── LangGraph ────────────────────────────────────────────────────
    subgraph LG["🤖  LangGraph StateGraph  (simulation/graph.py)"]
        direction TB
        ENTRY["Entry Node\nSeed battery SOC\nfrom scenario"]

        subgraph FAN["Parallel fan-out  ── 4 LLM agents ──"]
            direction LR
            DA["Demand Agent\ndemand_limit_kw\ndr_signal"]
            SA["Supply Agent\npv_divert_pct\ngrid_import_limit_kw"]
            BA["Battery Agent\ncharge_discharge_kw"]
            TA["Thermal Agent\nahu1/2_supply_c\nchiller_c · free_cool_pct"]
        end

        ORCH["Orchestration Agent\n(pure Python)\nReconcile → final_setpoints\nBuild agent_trace"]
    end

    %% ── Simulation ───────────────────────────────────────────────────
    subgraph SIM["⚡  EnergyPlus Engine  (simulation/engine.py)"]
        EP["5-zone Office\nCo-simulation\nSimulationEngine.run()"]
    end

    %% ── Battery ──────────────────────────────────────────────────────
    subgraph BAT["🔋  Battery Module  (simulation/battery.py)"]
        BS["BatteryState\n100 kWh capacity\n±25 kW  ·  C/2 taper >80% SOC"]
    end

    %% ── External ─────────────────────────────────────────────────────
    GROQ["☁️  Groq LLM API\nllama-3.3-70b-versatile\n(env-swappable via LLM_PROVIDER)"]

    %% ── Flows ────────────────────────────────────────────────────────
    UI_IN  -->|"POST /optimize\n{occupancy, ext_temp, pv_kw, tariff, scenario}"| OPT

    OPT    -->|BMSState| ENTRY
    ENTRY  --> DA & SA & BA & TA
    DA     -->|LLM call| GROQ
    SA     -->|LLM call| GROQ
    BA     -->|LLM call| GROQ
    TA     -->|LLM call| GROQ
    DA & SA & BA & TA -->|AgentAction\nproposed · score · rationale| ORCH

    ORCH   -->|Setpoints dataclass| EP
    EP     -->|SimResult\npower_kw · zone_temps\nenergy_forecast_kwh| OPT

    BA     -->|charge_discharge_kw| BS
    BS     -->|soc_pct\n(scenario-seeded)| ENTRY

    OPT    -->|"OptimizeResponse\n{metrics, setpoints, agent_trace, battery_soc_pct}"| UI_OUT

    %% ── Styles ───────────────────────────────────────────────────────
    classDef llm      fill:#7C3AED,color:#fff,stroke:#5B21B6
    classDef agent    fill:#1D4ED8,color:#fff,stroke:#1E40AF
    classDef infra    fill:#0F766E,color:#fff,stroke:#0D9488
    classDef battery  fill:#B45309,color:#fff,stroke:#92400E
    classDef external fill:#374151,color:#fff,stroke:#6B7280,stroke-dasharray:5 5

    class DA,SA,BA,TA agent
    class ORCH,ENTRY infra
    class EP infra
    class BS battery
    class GROQ external
```

## Component Summary

| Layer | File | Role |
|-------|------|------|
| **Dashboard** | `builmirai_mpc_hvac_dashboard.html` | Browser UI — sliders, metrics, agent graph panel, trace log, battery chip |
| **API** | `api/main.py` | FastAPI — `/optimize`, `/health`, `/scenarios`; wires graph → engine |
| **State schema** | `simulation/state.py` | `BMSState` + `AgentAction` TypedDicts shared across all nodes |
| **LangGraph graph** | `simulation/graph.py` | Entry node + 4 LLM agents (parallel fan-out) + Orchestration (fan-in) |
| **LLM factory** | `simulation/llm_provider.py` | `_get_llm()` — env-swappable via `LLM_PROVIDER` / `LLM_MODEL` |
| **Battery** | `simulation/battery.py` | SOC simulation, scenario seeds, C/2 taper above 80% |
| **EnergyPlus** | `simulation/engine.py` | pyenergyplus co-simulation, 5-zone office building |
| **Setpoints** | `simulation/mpc.py` | `Setpoints` dataclass — interface between graph and engine |
| **Models** | `api/models.py` | Pydantic request/response models incl. `AgentTraceEntry` |

## Request lifecycle

```
Browser click
  → POST /optimize (occupancy, ext_temp, pv_kw, tariff, scenario)
  → FastAPI builds BMSState
  → LangGraph: entry_node seeds battery SOC
  → [parallel] Demand · Supply · Battery · Thermal  →  Groq LLM calls
  → Orchestration reconciles → final_setpoints
  → EnergyPlus engine.run() → SimResult
  → Battery SOC updated
  → OptimizeResponse returned
  → Dashboard renders metrics, zones, forecast, agent trace, battery chip
```
