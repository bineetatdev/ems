import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("builmirai.api")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.models import AgentTraceEntry, OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse
from simulation.battery import get_battery, update_soc
from simulation.graph import graph
from simulation.mpc import Setpoints
from simulation.engine import SimulationEngine, EP_DIR
from simulation.state import BMSState

_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=True)


app = FastAPI(title="BuilMirai MPC API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ROOT = Path(__file__).parent.parent
_SIM_DIR = _ROOT / "simulation"
_IDF_PATH = _SIM_DIR / "building.idf"
_DASHBOARD = _ROOT / "builmirai_mpc_hvac_dashboard.html"


def _resolve_epw() -> Path:
    if os.environ.get("WEATHER_FILE"):
        return Path(os.environ["WEATHER_FILE"])
    bundled = _SIM_DIR / "weather.epw"
    if bundled.exists() and bundled.stat().st_size > 1000:
        return bundled
    ep_weather_dir = EP_DIR / "WeatherData"
    if ep_weather_dir.is_dir():
        epw_files = list(ep_weather_dir.glob("*.epw"))
        if epw_files:
            return epw_files[0]
    return bundled


_EPW_PATH = _resolve_epw()
engine = SimulationEngine(idf_path=_IDF_PATH, weather_path=_EPW_PATH, ep_dir=EP_DIR)
_engine_lock = asyncio.Lock()

SCENARIOS: dict[str, dict] = {
    "normal":   {"occupancy": 70,  "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0},
    "peak":     {"occupancy": 90,  "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0},
    "heatwave": {"occupancy": 75,  "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0},
    "preheat":  {"occupancy": 20,  "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0},
    "night":    {"occupancy": 5,   "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0},
}

ZONE_NAMES = ["Server Hall", "Open Plan", "Boardroom", "Reception", "Lab A"]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_DASHBOARD, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    epw_ok = _EPW_PATH.exists() and _EPW_PATH.stat().st_size > 1000
    return HealthResponse(
        status="ok" if epw_ok else "degraded",
        energyplus_version="25.2",
        idf_loaded=_IDF_PATH.exists(),
    )


@app.get("/scenarios/{name}", response_model=ScenarioResponse)
def get_scenario(name: str) -> ScenarioResponse:
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")
    return ScenarioResponse(name=name, **SCENARIOS[name])


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    battery = get_battery()
    initial_state: BMSState = {
        "occupancy": request.occupancy,
        "ext_temp": request.ext_temp,
        "pv_kw": request.pv_kw,
        "tariff": request.tariff,
        "net_power_kw": max(0.0, round(
            20.0 + (request.occupancy / 100.0) * 30.0
            + ((request.ext_temp - 10.0) / 32.0) * 18.0
            - request.pv_kw, 1
        )),
        "zone_temps": {z: 23.0 for z in ZONE_NAMES},
        "comfort_band": (22.0, 26.0),
        "battery_soc_pct": battery.soc_pct,
        "scenario": request.scenario,
        "demand_action": None,
        "supply_action": None,
        "battery_action": None,
        "thermal_action": None,
        "final_setpoints": None,
        "agent_trace": None,
    }

    loop = asyncio.get_running_loop()
    log.info("▶ graph.invoke starting  occ=%.0f ext=%.1f pv=%.1f tariff=%.1f",
             request.occupancy, request.ext_temp, request.pv_kw, request.tariff)
    t0 = time.perf_counter()
    try:
        result_state = await loop.run_in_executor(_executor, lambda: graph.invoke(initial_state))
    except Exception as exc:
        log.exception("graph.invoke FAILED after %.2fs", time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=f"Agent graph failed: {exc}") from exc
    log.info("✔ graph.invoke done in %.2fs", time.perf_counter() - t0)

    sp_dict = result_state["final_setpoints"]
    final_setpoints = Setpoints(
        ahu1_supply_c=sp_dict["ahu1_supply_c"],
        ahu2_supply_c=sp_dict["ahu2_supply_c"],
        chiller_c=sp_dict["chiller_c"],
        free_cool_pct=sp_dict["free_cool_pct"],
        pv_divert_pct=sp_dict["pv_divert_pct"],
        zone_cooling_sp_c=sp_dict["zone_cooling_sp_c"],
        zone_heating_sp_c=sp_dict["zone_heating_sp_c"],
        demand_limit_kw=sp_dict["demand_limit_kw"],
    )

    log.info("▶ engine.run starting")
    t1 = time.perf_counter()
    try:
        async with _engine_lock:
            sim_result = await loop.run_in_executor(
                _executor,
                lambda: engine.run(request=request, setpoints=final_setpoints),
            )
    except RuntimeError as exc:
        log.exception("engine.run FAILED after %.2fs", time.perf_counter() - t1)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc
    log.info("✔ engine.run done in %.2fs", time.perf_counter() - t1)

    bat_action = result_state.get("battery_action") or {}
    charge_kw = (bat_action.get("proposed") or {}).get("charge_discharge_kw", 0.0)
    new_soc = update_soc(battery.soc_pct, charge_kw)
    battery.soc_pct = new_soc

    raw_trace = result_state.get("agent_trace") or []
    agent_trace = [AgentTraceEntry(**entry) for entry in raw_trace]

    return OptimizeResponse(
        power_kw=sim_result.power_kw,
        savings_pct=sim_result.savings_pct,
        avg_zone_temp=sim_result.avg_zone_temp,
        pv_contribution_pct=sim_result.pv_contribution_pct,
        comfort_zones=sim_result.comfort_zones,
        zone_temps=sim_result.zone_temps,
        energy_forecast_kwh=sim_result.energy_forecast_kwh,
        setpoints=sim_result.setpoints,
        simulation_duration_s=sim_result.simulation_duration_s,
        battery_soc_pct=round(new_soc, 1),
        agent_trace=agent_trace,
    )
