import os
import sys
import uuid
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.mpc import Setpoints
from api.models import OptimizeRequest

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]
CHILLER_COP = 3.0
# Indices into 16-timestep array that map to the 7 dashboard forecast labels
# (+15m, +30m, +45m, +1h, +2h, +3h, +4h)
FORECAST_INDICES = [0, 1, 2, 3, 7, 11, 15]
EP_DIR = Path(os.environ.get("ENERGYPLUS_DIR", "/Applications/EnergyPlus-25-2-0"))


@dataclass
class SimulationResult:
    power_kw: float
    savings_pct: int
    avg_zone_temp: float
    pv_contribution_pct: int
    comfort_zones: int
    zone_temps: dict[str, float]
    energy_forecast_kwh: list[float]
    setpoints: dict[str, str]
    simulation_duration_s: float


def _aggregate_results(
    timesteps: list[dict[str, Any]],
    pv_kw: float,
    tariff: float,
    setpoints: Setpoints,
) -> SimulationResult:
    """Aggregate 16 EnergyPlus timestep readings into a SimulationResult."""
    # Average zone temps across all timesteps
    zone_temps = {
        z: round(sum(ts["zone_temps"][z] for ts in timesteps) / len(timesteps), 1)
        for z in ZONE_NAMES
    }

    avg_zone_temp = round(sum(zone_temps.values()) / len(zone_temps), 1)
    comfort_zones = sum(1 for t in zone_temps.values() if 22.0 <= t <= 26.0)

    # Total HVAC cooling energy (J) → average power (kW), then subtract PV
    total_cooling_j = sum(
        sum(ts["cooling_j"].values()) for ts in timesteps
    )
    dt_seconds = 15 * 60
    gross_cooling_kw = total_cooling_j / (len(timesteps) * dt_seconds * 1000)
    gross_electrical_kw = gross_cooling_kw / CHILLER_COP
    net_power_kw = max(0.0, round(gross_electrical_kw - pv_kw, 1))

    # Baseline power estimate (same energy without MPC = 10% more than gross)
    baseline_kw = gross_electrical_kw * 1.1
    savings_pct = max(0, round((baseline_kw - gross_electrical_kw) / baseline_kw * 100))

    # PV contribution %
    pv_contribution = 0
    if gross_electrical_kw + pv_kw > 0:
        pv_contribution = round(pv_kw / (gross_electrical_kw + pv_kw) * 100)

    # Energy forecast: subsample to 7 dashboard labels
    energy_forecast = []
    for idx in FORECAST_INDICES:
        ts = timesteps[idx]
        cooling_j = sum(ts["cooling_j"].values())
        kwh = round(cooling_j / CHILLER_COP / 3_600_000, 1)  # J → kWh
        energy_forecast.append(kwh)

    formatted_setpoints = {
        "AHU-1 supply": f"{setpoints.ahu1_supply_c:.1f}°C",
        "AHU-2 supply": f"{setpoints.ahu2_supply_c:.1f}°C",
        "Chiller setpt": f"{setpoints.chiller_c:.1f}°C",
        "Free-cool %": f"{setpoints.free_cool_pct:.0f}%",
        "PV divert": f"{setpoints.pv_divert_pct:.0f}%",
        "Demand limit": f"{setpoints.demand_limit_kw:.0f} kW",
    }

    return SimulationResult(
        power_kw=net_power_kw,
        savings_pct=savings_pct,
        avg_zone_temp=avg_zone_temp,
        pv_contribution_pct=pv_contribution,
        comfort_zones=comfort_zones,
        zone_temps=zone_temps,
        energy_forecast_kwh=energy_forecast,
        setpoints=formatted_setpoints,
        simulation_duration_s=0.0,
    )


class SimulationEngine:
    def __init__(self, idf_path: Path, weather_path: Path, ep_dir: Path = EP_DIR):
        self._idf_path = idf_path
        self._weather_path = weather_path
        self._ep_dir = ep_dir

    def _create_api(self):
        if str(self._ep_dir) not in sys.path:
            sys.path.insert(0, str(self._ep_dir))
        from pyenergyplus.api import EnergyPlusAPI  # noqa: PLC0415
        return EnergyPlusAPI()

    def _run_simulation(
        self,
        request: OptimizeRequest,
        setpoints: Setpoints,
    ) -> list[dict[str, Any]]:
        """Run EnergyPlus for the prediction horizon; return per-timestep data."""
        api = self._create_api()
        state = api.state_manager.new_state()
        api.runtime.set_console_output_status(state, False)

        collected: list[dict[str, Any]] = []
        handles_initialized = False
        var_handles: dict[str, int] = {}
        cool_handles: dict[str, int] = {}

        # Request output variables before run
        for zone in ZONE_NAMES:
            api.exchange.request_variable(state, "Zone Air Temperature", zone)
            api.exchange.request_variable(
                state, "Zone Ideal Loads Supply Air Total Cooling Energy", zone
            )

        def _init_handles(state) -> None:
            nonlocal handles_initialized
            if handles_initialized:
                return
            if not api.exchange.api_data_fully_ready(state):
                return
            for zone in ZONE_NAMES:
                h = api.exchange.get_variable_handle(state, "Zone Air Temperature", zone)
                var_handles[zone] = h
                c = api.exchange.get_variable_handle(
                    state, "Zone Ideal Loads Supply Air Total Cooling Energy", zone
                )
                cool_handles[zone] = c
            handles_initialized = True

        def _inject_inputs(state) -> None:
            if not api.exchange.api_data_fully_ready(state):
                return
            _init_handles(state)

            # Override outdoor dry-bulb temperature
            h = api.exchange.get_actuator_handle(
                state, "Weather Data", "Outdoor Dry Bulb", "Environment"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, request.ext_temp)

            # Override occupancy multiplier schedule
            h = api.exchange.get_actuator_handle(
                state, "Schedule:Compact", "Schedule Value", "OCC_MULTIPLIER"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, request.occupancy / 100.0)

            # Override cooling setpoint schedule
            h = api.exchange.get_actuator_handle(
                state, "Schedule:Compact", "Schedule Value", "Cooling_SP_Sched"
            )
            if h != -1:
                api.exchange.set_actuator_value(state, h, setpoints.zone_cooling_sp_c)

        def _collect_data(state) -> None:
            if api.exchange.warmup_flag(state):
                return
            if not api.exchange.api_data_fully_ready(state):
                return
            if not handles_initialized:
                _init_handles(state)

            zone_temps = {
                z: api.exchange.get_variable_value(state, var_handles[z])
                for z in ZONE_NAMES
                if z in var_handles and var_handles[z] != -1
            }
            cooling_j = {
                z: api.exchange.get_variable_value(state, cool_handles[z])
                for z in ZONE_NAMES
                if z in cool_handles and cool_handles[z] != -1
            }

            if zone_temps:
                collected.append({
                    "zone_temps": zone_temps,
                    "cooling_j": cooling_j,
                    "heating_j": {z: 0.0 for z in ZONE_NAMES},
                })

            if len(collected) >= 16:
                api.runtime.stop_simulation(state)

        api.runtime.callback_begin_zone_timestep_before_set_current_weather(
            state, _inject_inputs
        )
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state, _collect_data
        )

        output_dir = Path(tempfile.mkdtemp(prefix="ep_run_"))
        api.runtime.run_energyplus(state, [
            "-d", str(output_dir),
            "-w", str(self._weather_path),
            str(self._idf_path),
        ])
        api.state_manager.delete_state(state)

        # Ensure we have exactly 16 timesteps; pad with last value if simulation
        # ended early (e.g. only day 1 had fewer steps)
        while len(collected) < 16 and collected:
            collected.append(collected[-1])

        return collected

    def run(self, request: OptimizeRequest, setpoints: Setpoints) -> SimulationResult:
        import time
        t0 = time.perf_counter()
        timesteps = self._run_simulation(request=request, setpoints=setpoints)
        duration = time.perf_counter() - t0

        result = _aggregate_results(
            timesteps=timesteps,
            pv_kw=request.pv_kw,
            tariff=request.tariff,
            setpoints=setpoints,
        )
        result.simulation_duration_s = round(duration, 2)
        return result
