import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import OptimizeRequest, OptimizeResponse, HealthResponse, ScenarioResponse
from simulation.mpc import compute_setpoints
from simulation.engine import SimulationEngine, EP_DIR

app = FastAPI(title="BuilMirai MPC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_SIM_DIR = Path(__file__).parent.parent / "simulation"
_IDF_PATH = _SIM_DIR / "building.idf"
_EPW_PATH = _SIM_DIR / "weather.epw"

engine = SimulationEngine(idf_path=_IDF_PATH, weather_path=_EPW_PATH, ep_dir=EP_DIR)
_engine_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=1)

SCENARIOS: dict[str, dict] = {
    "normal":   {"occupancy": 70,  "ext_temp": 24.0, "pv_kw": 14.0, "tariff": 11.0},
    "peak":     {"occupancy": 90,  "ext_temp": 27.0, "pv_kw": 8.0,  "tariff": 34.0},
    "heatwave": {"occupancy": 75,  "ext_temp": 40.0, "pv_kw": 22.0, "tariff": 18.0},
    "preheat":  {"occupancy": 20,  "ext_temp": 19.0, "pv_kw": 5.0,  "tariff": 7.0},
    "night":    {"occupancy": 5,   "ext_temp": 16.0, "pv_kw": 0.0,  "tariff": 5.0},
}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
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
    setpoints = compute_setpoints(
        occupancy=request.occupancy,
        ext_temp=request.ext_temp,
        pv_kw=request.pv_kw,
        tariff=request.tariff,
    )
    loop = asyncio.get_event_loop()
    async with _engine_lock:
        result = await loop.run_in_executor(
            _executor, engine.run, request, setpoints
        )
    return OptimizeResponse(
        power_kw=result.power_kw,
        savings_pct=result.savings_pct,
        avg_zone_temp=result.avg_zone_temp,
        pv_contribution_pct=result.pv_contribution_pct,
        comfort_zones=result.comfort_zones,
        zone_temps=result.zone_temps,
        energy_forecast_kwh=result.energy_forecast_kwh,
        setpoints=result.setpoints,
        simulation_duration_s=result.simulation_duration_s,
    )
