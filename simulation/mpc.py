from dataclasses import dataclass


@dataclass
class Setpoints:
    ahu1_supply_c: float      # AHU-1 supply air temperature (°C)
    ahu2_supply_c: float      # AHU-2 supply air temperature (°C)
    chiller_c: float          # Chiller leaving water temp (°C)
    free_cool_pct: float      # Free-cooling fraction (0-100 %)
    pv_divert_pct: float      # PV self-consumption fraction (0-100 %)
    zone_cooling_sp_c: float  # Zone thermostat cooling setpoint (°C)
    zone_heating_sp_c: float  # Zone thermostat heating setpoint (°C)
    demand_limit_kw: float    # Demand limit (kW)


# Base setpoints before rule adjustments
_BASE_AHU1 = 18.0
_BASE_AHU2 = 18.0
_BASE_CHILLER = 7.0
_BASE_FREE_COOL = 30.0
_BASE_COOLING_SP = 25.0
_BASE_HEATING_SP = 20.0
_BASE_DEMAND_LIMIT = 80.0


def compute_setpoints(
    occupancy: float,  # 0-100 %
    ext_temp: float,   # °C
    pv_kw: float,      # kW
    tariff: float,     # p/kWh
) -> Setpoints:
    """Compute HVAC setpoints using heuristic MPC rules.

    Rules combine additively from base values. Clamping to physical
    bounds is applied at the end.
    """
    ahu1 = _BASE_AHU1
    ahu2 = _BASE_AHU2
    chiller = _BASE_CHILLER
    free_cool = _BASE_FREE_COOL
    cooling_sp = _BASE_COOLING_SP
    heating_sp = _BASE_HEATING_SP
    demand_limit = _BASE_DEMAND_LIMIT

    # Rule 1: High tariff → reduce cooling load (raise supply temps)
    if tariff > 25:
        delta = (tariff - 25) / 15 * 2.0  # up to +2°C at max tariff
        ahu1 += delta
        ahu2 += delta

    # Rule 2: High PV → exploit free energy (lower chiller, more free-cooling)
    if pv_kw > 15:
        delta_chiller = (pv_kw - 15) / 15 * 1.0  # up to -1°C
        chiller -= delta_chiller
        free_cool += (pv_kw - 15) / 15 * 20.0  # up to +20%

    # Rule 3: Heatwave → pre-cool aggressively, tighten demand limit
    if ext_temp > 35:
        free_cool += (ext_temp - 35) / 7 * 30.0  # up to +30%
        demand_limit -= (ext_temp - 35) / 7 * 20.0  # tighten by up to 20 kW

    # Rule 4: Low occupancy (night setback) → relax zone setpoints
    if occupancy < 20:
        relax = (20 - occupancy) / 20 * 1.5  # up to +1.5°C
        cooling_sp += relax
        heating_sp -= relax / 2

    # Rule 5: Dense occupancy → tighten zone setpoints for comfort
    if occupancy > 80:
        tighten = (occupancy - 80) / 20 * 1.0  # up to -1°C
        cooling_sp -= tighten

    # PV divert scales linearly with PV output
    pv_divert = min(100.0, (pv_kw / 30) * 100)

    return Setpoints(
        ahu1_supply_c=max(12.0, min(24.0, ahu1)),
        ahu2_supply_c=max(12.0, min(24.0, ahu2)),
        chiller_c=max(4.0, min(12.0, chiller)),
        free_cool_pct=max(0.0, min(100.0, free_cool)),
        pv_divert_pct=max(0.0, min(100.0, pv_divert)),
        zone_cooling_sp_c=max(22.0, min(28.0, cooling_sp)),
        zone_heating_sp_c=max(16.0, min(22.0, heating_sp)),
        demand_limit_kw=max(20.0, min(200.0, demand_limit)),
    )
