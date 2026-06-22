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
