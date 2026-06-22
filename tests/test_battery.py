import pytest
from simulation.battery import update_soc, SCENARIO_SOC, get_battery, reset_battery


def test_charging_increases_soc():
    soc = update_soc(50.0, charge_discharge_kw=20.0, dt_minutes=15)
    # 20 kW * 0.25 h = 5 kWh into 100 kWh battery → +5%
    assert abs(soc - 55.0) < 0.1


def test_discharging_decreases_soc():
    soc = update_soc(60.0, charge_discharge_kw=-15.0, dt_minutes=15)
    # 15 kW * 0.25 h = 3.75 kWh out → -3.75%
    assert abs(soc - 56.25) < 0.1


def test_soc_clamped_to_100():
    soc = update_soc(98.0, charge_discharge_kw=25.0, dt_minutes=15)
    assert soc == 100.0


def test_soc_clamped_to_0():
    soc = update_soc(2.0, charge_discharge_kw=-25.0, dt_minutes=15)
    assert soc == 0.0


def test_charge_rate_tapers_above_80_pct():
    # Above 80% SOC, max charge rate drops to 12.5 kW
    soc_tapered = update_soc(85.0, charge_discharge_kw=25.0, dt_minutes=15)
    # At 85% with taper: 12.5 kW * 0.25h = 3.125% added
    assert abs(soc_tapered - 88.125) < 0.1


def test_scenario_soc_seeds():
    assert SCENARIO_SOC["normal"] == 50.0
    assert SCENARIO_SOC["peak"] == 60.0
    assert SCENARIO_SOC["heatwave"] == 30.0
    assert SCENARIO_SOC["preheat"] == 80.0
    assert SCENARIO_SOC["night"] == 90.0


def test_singleton_persists_across_updates():
    reset_battery(50.0)
    bat = get_battery()
    assert bat.soc_pct == 50.0
