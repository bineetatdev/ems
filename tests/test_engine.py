import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from simulation.engine import SimulationEngine, SimulationResult, _aggregate_results
from simulation.mpc import Setpoints
from api.models import OptimizeRequest


FAKE_SETPOINTS = Setpoints(
    ahu1_supply_c=18.0, ahu2_supply_c=18.0, chiller_c=7.0,
    free_cool_pct=30.0, pv_divert_pct=47.0,
    zone_cooling_sp_c=25.0, zone_heating_sp_c=20.0, demand_limit_kw=80.0,
)

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]

FAKE_TIMESTEPS = [
    {"zone_temps": {z: 23.0 + i * 0.1 for i, z in enumerate(ZONE_NAMES)},
     "cooling_j": {z: 10000.0 for z in ZONE_NAMES},
     "heating_j": {z: 0.0 for z in ZONE_NAMES}}
    for _ in range(16)
]


def test_aggregate_results_returns_simulation_result():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert isinstance(result, SimulationResult)


def test_aggregate_results_zone_temps_averaged():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert "Server Hall" in result.zone_temps
    assert "Lab A" in result.zone_temps
    assert 20.0 <= result.zone_temps["Server Hall"] <= 30.0


def test_aggregate_results_energy_forecast_has_7_points():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert len(result.energy_forecast_kwh) == 7


def test_aggregate_results_comfort_zones_counts_in_band():
    # All zones at ~23°C → should be in 22-26 band → comfort_zones = 5
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result.comfort_zones == 5


def test_aggregate_results_power_reduced_by_pv():
    result_no_pv = _aggregate_results(FAKE_TIMESTEPS, pv_kw=0.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    result_with_pv = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result_with_pv.power_kw <= result_no_pv.power_kw


def test_aggregate_results_savings_pct_non_negative():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert result.savings_pct >= 0


def test_aggregate_results_setpoints_formatted():
    result = _aggregate_results(FAKE_TIMESTEPS, pv_kw=14.0, tariff=11.0, setpoints=FAKE_SETPOINTS)
    assert "AHU-1 supply" in result.setpoints
    assert "°C" in result.setpoints["AHU-1 supply"]
    assert "%" in result.setpoints["Free-cool %"]


def test_engine_run_calls_run_simulation(tmp_path):
    idf = tmp_path / "test.idf"
    idf.write_text("")
    epw = tmp_path / "test.epw"
    epw.write_text("")

    engine = SimulationEngine(
        idf_path=idf,
        weather_path=epw,
        ep_dir=Path("/Applications/EnergyPlus-25-2-0"),
    )

    request = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)

    with patch.object(engine, "_run_simulation", return_value=FAKE_TIMESTEPS) as mock_sim:
        result = engine.run(request, FAKE_SETPOINTS)

    mock_sim.assert_called_once()
    assert isinstance(result, SimulationResult)


def test_engine_run_passes_setpoints_to_simulation(tmp_path):
    idf = tmp_path / "test.idf"
    idf.write_text("")
    epw = tmp_path / "test.epw"
    epw.write_text("")

    engine = SimulationEngine(idf_path=idf, weather_path=epw,
                              ep_dir=Path("/Applications/EnergyPlus-25-2-0"))
    request = OptimizeRequest(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)

    with patch.object(engine, "_run_simulation", return_value=FAKE_TIMESTEPS) as mock_sim:
        engine.run(request, FAKE_SETPOINTS)

    _, kwargs = mock_sim.call_args
    assert kwargs["setpoints"] == FAKE_SETPOINTS
