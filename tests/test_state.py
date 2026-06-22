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
        "agent_trace": None,
    }
    assert state["comfort_band"] == (22.0, 26.0)
    assert state["demand_action"] is None
