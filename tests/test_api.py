import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from simulation.engine import SimulationResult
from simulation.mpc import Setpoints


FAKE_RESULT = SimulationResult(
    power_kw=38.2, savings_pct=18, avg_zone_temp=23.4, pv_contribution_pct=37,
    comfort_zones=5,
    zone_temps={"Server Hall": 22.1, "Open Plan": 23.4, "Boardroom": 23.8,
                "Reception": 22.9, "Lab A": 23.6},
    energy_forecast_kwh=[38.0, 35.0, 33.0, 31.0, 28.0, 26.0, 24.0],
    setpoints={"AHU-1 supply": "17.8°C", "AHU-2 supply": "18.2°C",
               "Chiller setpt": "6.8°C", "Free-cool %": "45%",
               "PV divert": "47%", "Demand limit": "46 kW"},
    simulation_duration_s=12.4,
)

FAKE_GRAPH_STATE = {
    "final_setpoints": {
        "ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0,
        "chiller_c": 7.0, "free_cool_pct": 30.0,
        "pv_divert_pct": 47.0, "zone_cooling_sp_c": 24.0,
        "zone_heating_sp_c": 19.0, "demand_limit_kw": 80.0,
    },
    "demand_action": {"proposed": {"demand_limit_kw": 80.0, "dr_signal": "normal"}, "score": 0.7, "rationale": "ok"},
    "supply_action": {"proposed": {"pv_divert_pct": 47.0, "grid_import_limit_kw": 90.0}, "score": 0.6, "rationale": "ok"},
    "battery_action": {"proposed": {"charge_discharge_kw": 0.0}, "score": 0.5, "rationale": "ok"},
    "thermal_action": {"proposed": {"ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 30.0}, "score": 1.0, "rationale": "ok"},
    "agent_trace": [],
    "battery_soc_pct": 50.0,
}


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "energyplus_version" in data
    assert "idf_loaded" in data


def test_optimize_returns_200_with_valid_input(client):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = FAKE_GRAPH_STATE
        mock_engine.run.return_value = FAKE_RESULT
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    assert resp.status_code == 200


def test_optimize_response_has_all_fields(client):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = FAKE_GRAPH_STATE
        mock_engine.run.return_value = FAKE_RESULT
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    data = resp.json()
    assert "power_kw" in data
    assert "zone_temps" in data
    assert "energy_forecast_kwh" in data
    assert "setpoints" in data
    assert len(data["energy_forecast_kwh"]) == 7
    assert "Server Hall" in data["zone_temps"]


def test_optimize_rejects_invalid_occupancy(client):
    resp = client.post("/optimize", json={
        "occupancy": 150, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
    })
    assert resp.status_code == 422


def test_optimize_rejects_missing_field(client):
    resp = client.post("/optimize", json={
        "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0
        # tariff missing
    })
    assert resp.status_code == 422


def test_scenarios_normal(client):
    resp = client.get("/scenarios/normal")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "normal"
    assert data["occupancy"] == 70
    assert data["ext_temp"] == 24.0


def test_scenarios_peak(client):
    resp = client.get("/scenarios/peak")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tariff"] == 34.0


def test_scenarios_unknown_returns_404(client):
    resp = client.get("/scenarios/nonexistent")
    assert resp.status_code == 404


def test_optimize_calls_graph_then_engine(client):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = FAKE_GRAPH_STATE
        mock_engine.run.return_value = FAKE_RESULT

        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })

    assert resp.status_code == 200
    mock_graph.invoke.assert_called_once()
    mock_engine.run.assert_called_once()


def test_optimize_handles_engine_failure(client):
    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = FAKE_GRAPH_STATE
        mock_engine.run.side_effect = RuntimeError("EnergyPlus crashed")
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    assert resp.status_code == 500
    assert "Simulation failed" in resp.json()["detail"]


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
    call_args = mock_graph.invoke.call_args[0][0]
    assert call_args["scenario"] == "night"


@pytest.mark.parametrize("scenario,payload,min_comfort", [
    ("normal",   {"occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0, "scenario": "normal"},   3),
    ("peak",     {"occupancy": 90, "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0, "scenario": "peak"},     3),
    ("heatwave", {"occupancy": 75, "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0, "scenario": "heatwave"}, 3),
    ("preheat",  {"occupancy": 20, "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0,  "scenario": "preheat"},  3),
    ("night",    {"occupancy": 5,  "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0,  "scenario": "night"},    3),
])
def test_scenario_smoke(client, scenario, payload, min_comfort):
    from unittest.mock import patch
    from simulation.engine import SimulationResult

    mock_graph_state = {
        "final_setpoints": {
            "ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0,
            "chiller_c": 7.0, "free_cool_pct": 30.0,
            "pv_divert_pct": 46.7, "zone_cooling_sp_c": 24.0,
            "zone_heating_sp_c": 19.0, "demand_limit_kw": 80.0,
        },
        "demand_action":  {"proposed": {"demand_limit_kw": 80.0, "dr_signal": "normal"}, "score": 0.7, "rationale": "ok"},
        "supply_action":  {"proposed": {"pv_divert_pct": 46.7, "grid_import_limit_kw": 90.0}, "score": 0.6, "rationale": "ok"},
        "battery_action": {"proposed": {"charge_discharge_kw": 0.0}, "score": 0.5, "rationale": "ok"},
        "thermal_action": {"proposed": {"ahu1_supply_c": 18.0, "ahu2_supply_c": 18.0, "chiller_c": 7.0, "free_cool_pct": 30.0}, "score": 1.0, "rationale": "ok"},
        "agent_trace": [
            {"agent": "demand", "status": "accepted", "proposed": {}, "score": 0.7, "rationale": "ok"},
            {"agent": "supply", "status": "accepted", "proposed": {}, "score": 0.6, "rationale": "ok"},
            {"agent": "battery", "status": "accepted", "proposed": {}, "score": 0.5, "rationale": "ok"},
            {"agent": "thermal", "status": "accepted", "proposed": {}, "score": 1.0, "rationale": "ok"},
            {"agent": "orchestration", "status": "—", "proposed": {}, "score": 1.0, "rationale": "Reconciled"},
        ],
        "battery_soc_pct": 50.0,
    }

    mock_sim = SimulationResult(
        power_kw=30.0, savings_pct=10, avg_zone_temp=23.5,
        pv_contribution_pct=40, comfort_zones=min_comfort,
        zone_temps={"Server Hall": 23.5, "Open Plan": 24.0, "Boardroom": 22.8,
                    "Reception": 23.1, "Lab A": 24.5},
        energy_forecast_kwh=[1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8],
        setpoints={"AHU-1 supply": "18.0°C", "AHU-2 supply": "18.0°C",
                   "Chiller setpt": "7.0°C", "Free-cool %": "30%",
                   "PV divert": "47%", "Demand limit": "80 kW"},
        simulation_duration_s=4.2,
    )

    with patch("api.main.graph") as mock_graph, \
         patch("api.main.engine") as mock_engine:
        mock_graph.invoke.return_value = mock_graph_state
        mock_engine.run.return_value = mock_sim

        resp = client.post("/optimize", json=payload)

    assert resp.status_code == 200, f"{scenario}: {resp.text}"
    data = resp.json()
    assert data["comfort_zones"] >= min_comfort, f"{scenario}: comfort_zones={data['comfort_zones']}"
    assert len(data["agent_trace"]) == 5, f"{scenario}: trace length {len(data['agent_trace'])}"
    sp = data["setpoints"]
    assert "AHU-1 supply" in sp, f"{scenario}: missing AHU-1 supply"
    assert "Chiller setpt" in sp, f"{scenario}: missing Chiller setpt"
    assert "Demand limit" in sp, f"{scenario}: missing Demand limit"
