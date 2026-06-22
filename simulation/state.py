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
    agent_trace: list | None
