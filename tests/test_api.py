import pytest
from unittest.mock import patch, MagicMock
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
    with patch("api.main.engine") as mock_engine:
        mock_engine.run.return_value = FAKE_RESULT
        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })
    assert resp.status_code == 200


def test_optimize_response_has_all_fields(client):
    with patch("api.main.engine") as mock_engine:
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


def test_optimize_calls_compute_setpoints_then_engine(client):
    with patch("api.main.compute_setpoints") as mock_mpc, \
         patch("api.main.engine") as mock_engine:
        fake_sp = Setpoints(ahu1_supply_c=18.0, ahu2_supply_c=18.0, chiller_c=7.0,
                            free_cool_pct=30.0, pv_divert_pct=47.0,
                            zone_cooling_sp_c=25.0, zone_heating_sp_c=20.0,
                            demand_limit_kw=80.0)
        mock_mpc.return_value = fake_sp
        mock_engine.run.return_value = FAKE_RESULT

        resp = client.post("/optimize", json={
            "occupancy": 70, "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0
        })

    mock_mpc.assert_called_once_with(
        occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0
    )
    mock_engine.run.assert_called_once()
