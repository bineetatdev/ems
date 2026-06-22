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
