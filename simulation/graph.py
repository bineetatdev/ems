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

    thermal_status = "accepted"
    trace = [
        {"agent": "demand",  "status": "accepted", "proposed": demand["proposed"],  "score": demand["score"],  "rationale": demand["rationale"]},
        {"agent": "supply",  "status": "accepted", "proposed": supply["proposed"],  "score": supply["score"],  "rationale": supply["rationale"]},
        {"agent": "battery", "status": "accepted", "proposed": battery["proposed"], "score": battery["score"], "rationale": battery["rationale"]},
        {"agent": "thermal", "status": thermal_status, "proposed": thermal["proposed"], "score": thermal["score"], "rationale": thermal["rationale"]},
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
