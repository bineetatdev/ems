import json
from unittest.mock import MagicMock, patch
from simulation.state import AgentAction, BMSState


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
        "agent_trace": None,
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


def test_orchestration_applies_comfort_hard_constraint():
    from simulation.graph import orchestration_agent
    state = _make_state(
        zone_temps={"Server Hall": 27.5, "Open Plan": 28.0, "Boardroom": 23.0,
                    "Reception": 22.5, "Lab A": 27.0},
        comfort_band=(22.0, 26.0),
        demand_action={"proposed": {"demand_limit_kw": 80.0, "dr_signal": "normal"}, "score": 0.7, "rationale": "ok"},
        supply_action={"proposed": {"pv_divert_pct": 60.0, "grid_import_limit_kw": 100.0}, "score": 0.6, "rationale": "ok"},
        battery_action={"proposed": {"charge_discharge_kw": -10.0}, "score": 0.5, "rationale": "ok"},
        thermal_action={
            "proposed": {"ahu1_supply_c": 16.0, "ahu2_supply_c": 16.0, "chiller_c": 5.0, "free_cool_pct": 10.0},
            "score": 0.4,
            "rationale": "Aggressive cooling required",
        },
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
        demand_action={"proposed": {"demand_limit_kw": 75.0, "dr_signal": "curtail"}, "score": 0.8, "rationale": "ok"},
        supply_action={"proposed": {"pv_divert_pct": 80.0, "grid_import_limit_kw": 60.0}, "score": 0.75, "rationale": "ok"},
        battery_action={"proposed": {"charge_discharge_kw": 15.0}, "score": 0.9, "rationale": "ok"},
        thermal_action={
            "proposed": {"ahu1_supply_c": 18.5, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 40.0},
            "score": 1.0,
            "rationale": "All zones in band",
        },
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

    # LangGraph runs fan-out agents (demand/supply/battery/thermal) in parallel
    # threads, so the call order to _get_llm is non-deterministic.
    # Use a single omnibus mock payload that satisfies all 4 agent parsers.
    omnibus_mock = _mock_llm_response({
        "demand_limit_kw": 78.0, "dr_signal": "normal",
        "pv_divert_pct": 47.0, "grid_import_limit_kw": 90.0,
        "charge_discharge_kw": 5.0,
        "ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0,
        "chiller_c": 7.0, "free_cool_pct": 30.0,
        "score": 0.7, "rationale": "Normal conditions",
    })

    with patch("simulation.graph._get_llm", return_value=omnibus_mock):
        result_state = graph.invoke(state)

    assert result_state["final_setpoints"] is not None
    assert result_state["demand_action"] is not None
    assert result_state["supply_action"] is not None
    assert result_state["battery_action"] is not None
    assert result_state["thermal_action"] is not None
