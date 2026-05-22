from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    occupancy: float = Field(..., ge=0, le=100)
    ext_temp: float = Field(..., ge=10, le=42)
    pv_kw: float = Field(..., ge=0, le=30)
    tariff: float = Field(..., ge=5, le=40)
    horizon_hours: int = Field(default=4, ge=1, le=8)


class OptimizeResponse(BaseModel):
    power_kw: float
    savings_pct: int
    avg_zone_temp: float
    pv_contribution_pct: int
    comfort_zones: int
    zone_temps: dict[str, float]
    energy_forecast_kwh: list[float]
    setpoints: dict[str, str]
    simulation_duration_s: float


class HealthResponse(BaseModel):
    status: str
    energyplus_version: str
    idf_loaded: bool


class ScenarioResponse(BaseModel):
    name: str
    occupancy: float
    ext_temp: float
    pv_kw: float
    tariff: float
