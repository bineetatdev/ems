from dataclasses import dataclass


SCENARIO_SOC: dict[str, float] = {
    "normal":   50.0,
    "peak":     60.0,
    "heatwave": 30.0,
    "preheat":  80.0,
    "night":    90.0,
}

_CAPACITY_KWH = 100.0
_MAX_RATE_KW = 25.0
_TAPER_THRESHOLD_PCT = 80.0
_TAPERED_RATE_KW = 12.5  # C/2 taper above threshold


@dataclass
class BatteryState:
    capacity_kwh: float = _CAPACITY_KWH
    max_rate_kw: float = _MAX_RATE_KW
    soc_pct: float = 50.0


_battery = BatteryState()


def get_battery() -> BatteryState:
    return _battery


def reset_battery(soc_pct: float) -> None:
    _battery.soc_pct = soc_pct


def update_soc(
    current_pct: float,
    charge_discharge_kw: float,
    dt_minutes: float = 15.0,
) -> float:
    """Return new SOC % after applying charge_discharge_kw for dt_minutes.

    Positive kW = charging, negative = discharging.
    Applies C/2 taper above 80% SOC on the charge side.
    """
    effective_kw = charge_discharge_kw
    if charge_discharge_kw > 0 and current_pct >= _TAPER_THRESHOLD_PCT:
        effective_kw = min(charge_discharge_kw, _TAPERED_RATE_KW)

    dt_hours = dt_minutes / 60.0
    delta_pct = (effective_kw * dt_hours / _CAPACITY_KWH) * 100.0
    new_pct = current_pct + delta_pct
    return max(0.0, min(100.0, new_pct))
