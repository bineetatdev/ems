import pytest
from simulation.mpc import Setpoints, compute_setpoints


BASE = dict(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)


def test_returns_setpoints_instance():
    result = compute_setpoints(**BASE)
    assert isinstance(result, Setpoints)


def test_base_setpoints():
    """Default inputs produce base setpoints."""
    sp = compute_setpoints(**BASE)
    assert 16.0 <= sp.ahu1_supply_c <= 22.0
    assert 16.0 <= sp.ahu2_supply_c <= 22.0
    assert 5.0 <= sp.chiller_c <= 10.0
    assert 0.0 <= sp.free_cool_pct <= 100.0
    assert 0.0 <= sp.pv_divert_pct <= 100.0


def test_high_tariff_raises_supply_temps():
    """Peak tariff → raise supply air temps to reduce cooling load."""
    low = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=10.0)
    high = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=35.0)
    assert high.ahu1_supply_c > low.ahu1_supply_c
    assert high.ahu2_supply_c > low.ahu2_supply_c


def test_high_pv_lowers_chiller_setpoint():
    """High PV generation → lower chiller setpoint (exploit free energy)."""
    low_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=2.0, tariff=11.0)
    high_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=25.0, tariff=11.0)
    assert high_pv.chiller_c < low_pv.chiller_c


def test_high_pv_increases_free_cool():
    """High PV → increase free cooling fraction."""
    low_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=2.0, tariff=11.0)
    high_pv = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=25.0, tariff=11.0)
    assert high_pv.free_cool_pct > low_pv.free_cool_pct


def test_heatwave_increases_free_cool_and_tightens_demand():
    """Heatwave (ext_temp > 35) → maximise pre-cooling."""
    normal = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    heatwave = compute_setpoints(occupancy=70, ext_temp=40.0, pv_kw=14.0, tariff=11.0)
    assert heatwave.demand_limit_kw < normal.demand_limit_kw


def test_low_occupancy_raises_setpoints():
    """Night setback (occupancy < 20) → relax zone setpoints."""
    normal = compute_setpoints(occupancy=70, ext_temp=24.0, pv_kw=5.0, tariff=11.0)
    night = compute_setpoints(occupancy=5, ext_temp=24.0, pv_kw=0.0, tariff=5.0)
    assert night.zone_cooling_sp_c > normal.zone_cooling_sp_c


def test_high_occupancy_tightens_setpoints():
    """Dense occupancy → tighten zone setpoints for comfort."""
    low = compute_setpoints(occupancy=20, ext_temp=24.0, pv_kw=5.0, tariff=11.0)
    high = compute_setpoints(occupancy=95, ext_temp=24.0, pv_kw=14.0, tariff=11.0)
    assert high.zone_cooling_sp_c <= low.zone_cooling_sp_c


def test_setpoints_stay_in_physical_bounds():
    """Setpoints never leave physically valid ranges regardless of inputs."""
    for occ in [0, 50, 100]:
        for ext in [10, 30, 42]:
            for pv in [0, 15, 30]:
                for tariff in [5, 20, 40]:
                    sp = compute_setpoints(occ, ext, pv, tariff)
                    assert 12.0 <= sp.ahu1_supply_c <= 24.0, f"AHU1 out of range: {sp.ahu1_supply_c}"
                    assert 12.0 <= sp.ahu2_supply_c <= 24.0
                    assert 4.0 <= sp.chiller_c <= 12.0
                    assert 0 <= sp.free_cool_pct <= 100
                    assert 0 <= sp.pv_divert_pct <= 100
                    assert 20 <= sp.demand_limit_kw <= 200
