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
