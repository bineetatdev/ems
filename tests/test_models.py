import pytest
from pydantic import ValidationError
from api.models import OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse


def test_optimize_request_valid():
    req = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    assert req.occupancy == 70
    assert req.horizon_hours == 4  # default


def test_optimize_request_defaults_horizon():
    req = OptimizeRequest(occupancy=50, ext_temp=20.0, pv_kw=5.0, tariff=10.0)
    assert req.horizon_hours == 4


def test_optimize_request_rejects_out_of_range_occupancy():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=101, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_negative_occupancy():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=-1, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_ext_temp_too_high():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=43.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_ext_temp_too_low():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=9.0, pv_kw=14.0, tariff=11.0)


def test_optimize_request_rejects_pv_out_of_range():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=31.0, tariff=11.0)


def test_optimize_request_rejects_tariff_out_of_range():
    with pytest.raises(ValidationError):
        OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=41.0)


def test_optimize_response_has_all_fields():
    resp = OptimizeResponse(
        power_kw=38.2,
        savings_pct=18,
        avg_zone_temp=23.4,
        pv_contribution_pct=37,
        comfort_zones=5,
        zone_temps={"Server Hall": 22.1, "Open Plan": 23.4, "Boardroom": 23.8, "Reception": 22.9, "Lab A": 23.6},
        energy_forecast_kwh=[38, 35, 33, 31, 28, 26, 24],
        setpoints={"AHU-1 supply": "17.8°C", "AHU-2 supply": "18.2°C", "Chiller setpt": "6.8°C",
                   "Free-cool %": "45%", "PV divert": "47%", "Demand limit": "46 kW"},
        simulation_duration_s=12.4,
    )
    assert resp.comfort_zones == 5
    assert len(resp.energy_forecast_kwh) == 7
    assert "Server Hall" in resp.zone_temps


def test_health_response():
    h = HealthResponse(status="ok", energyplus_version="25.2", idf_loaded=True)
    assert h.status == "ok"


def test_scenario_response():
    s = ScenarioResponse(name="peak", occupancy=90, ext_temp=27.0, pv_kw=8.0, tariff=34.0)
    assert s.name == "peak"
