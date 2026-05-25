"""Generate simulation/building.idf for the 5-zone BuilMirai office building."""

from dataclasses import dataclass
from pathlib import Path

ENERGYPLUS_VERSION = "25.2.0"
H = 3.5  # zone height (m)
OUTPUT_PATH = Path(__file__).parent / "building.idf"


@dataclass
class ZoneSpec:
    name: str
    width: float   # x extent (m)
    depth: float   # y extent (north-south) (m)
    x0: float      # x origin (zones are spaced along x-axis to avoid overlap)
    people: int    # number of occupants at full occupancy
    equip_wm2: float  # electric equipment W/m²
    lighting_wm2: float = 10.0
    server_room: bool = False  # 24/7 equipment, no people


ZONES = [
    ZoneSpec("Server Hall",  width=5,  depth=6,  x0=0,   people=0,   equip_wm2=100.0, server_room=True),
    ZoneSpec("Open Plan",    width=14, depth=14, x0=10,  people=40,  equip_wm2=10.0),
    ZoneSpec("Boardroom",    width=7,  depth=7,  x0=30,  people=20,  equip_wm2=5.0),
    ZoneSpec("Reception",    width=6,  depth=7,  x0=42,  people=5,   equip_wm2=5.0),
    ZoneSpec("Lab A",        width=9,  depth=9,  x0=53,  people=15,  equip_wm2=40.0),
]


def _wall(name: str, zone: str, x1: float, y1: float, x2: float, y2: float,
          z_lo: float, z_hi: float, face: str, bc: str = "Adiabatic") -> str:
    """Generate a BuildingSurface:Detailed wall entry.

    face: 'S', 'N', 'E', 'W', 'Floor', 'Ceiling'
    Vertex order: upper-left first, CCW from outside (EnergyPlus convention).
    """
    sun = "SunExposed" if bc == "Outdoors" else "NoSun"
    wind = "WindExposed" if bc == "Outdoors" else "NoWind"

    if face == "S":   # y=y1, outward normal = -y
        verts = f"  {x1},{y1},{z_hi},\n  {x1},{y1},{z_lo},\n  {x2},{y1},{z_lo},\n  {x2},{y1},{z_hi}"
    elif face == "N": # y=y2, outward normal = +y
        verts = f"  {x2},{y2},{z_hi},\n  {x2},{y2},{z_lo},\n  {x1},{y2},{z_lo},\n  {x1},{y2},{z_hi}"
    elif face == "E": # x=x2, outward normal = +x
        verts = f"  {x2},{y1},{z_hi},\n  {x2},{y1},{z_lo},\n  {x2},{y2},{z_lo},\n  {x2},{y2},{z_hi}"
    elif face == "W": # x=x1, outward normal = -x
        verts = f"  {x1},{y2},{z_hi},\n  {x1},{y2},{z_lo},\n  {x1},{y1},{z_lo},\n  {x1},{y1},{z_hi}"
    elif face == "Floor":
        verts = f"  {x1},{y1},{z_lo},\n  {x1},{y2},{z_lo},\n  {x2},{y2},{z_lo},\n  {x2},{y1},{z_lo}"
    elif face == "Ceiling":
        verts = f"  {x1},{y2},{z_hi},\n  {x2},{y2},{z_hi},\n  {x2},{y1},{z_hi},\n  {x1},{y1},{z_hi}"
    else:
        raise ValueError(f"Unknown face: {face}")

    surface_type = "Floor" if face == "Floor" else ("Ceiling" if face == "Ceiling" else "Wall")
    bc_cond = bc if bc != "Adiabatic" else "Adiabatic"

    return f"""BuildingSurface:Detailed,
  {name},                   !- Name
  {surface_type},           !- Surface Type
  {"Floor_Const" if face == "Floor" else "ExtWall_Const" if bc == "Outdoors" else "Adiabatic_Const"},
  {zone},                   !- Zone Name
  ,                         !- Space Name
  {bc_cond},               !- Outside Boundary Condition
  {"" if bc != "Outdoors" else ""},  !- Outside Boundary Condition Object
  {sun},                    !- Sun Exposure
  {wind},                   !- Wind Exposure
  {"0.5" if face == "Floor" else "autocalculate"},
  4,
{verts};
"""


def _window(name: str, parent_wall: str, zone: str,
            x1: float, x2: float, y: float, z_lo: float, z_hi: float) -> str:
    """Window on south-facing wall (y=const, x1→x2)."""
    wx1 = x1 + (x2 - x1) * 0.1
    wx2 = x2 - (x2 - x1) * 0.1
    wz_lo = z_lo + 0.8
    wz_hi = z_hi - 0.5
    return f"""FenestrationSurface:Detailed,
  {name},
  Window,
  Window_Const,             !- Construction Name
  {parent_wall},            !- Building Surface Name
  ,  ,  ,  ,
  4,
  {wx1},{y},{wz_hi},
  {wx1},{y},{wz_lo},
  {wx2},{y},{wz_lo},
  {wx2},{y},{wz_hi};
"""


def _zone_hvac(z: ZoneSpec) -> str:
    name = z.name
    return f"""
ZoneHVAC:IdealLoadsAirSystem,
  {name}_IdealLoads,        !- Name
  ,                         !- Availability Schedule
  {name}_SupplyAirInlet,    !- Zone Supply Air Node Name
  {name}_ExhaustAirOutlet,  !- Zone Exhaust Air Node Name
  ,                         !- System Inlet Air Node Name
  50,                       !- Maximum Heating Supply Air Temperature {{C}}
  13,                       !- Minimum Cooling Supply Air Temperature {{C}}
  0.015,                    !- Maximum Heating Supply Air Humidity Ratio {{kgWater/kgDryAir}}
  0.009,                    !- Minimum Cooling Supply Air Humidity Ratio {{kgWater/kgDryAir}}
  NoLimit,                  !- Heating Limit
  ,                         !- Maximum Heating Air Flow Rate (autosize)
  ,                         !- Maximum Sensible Heating Capacity (autosize)
  NoLimit,                  !- Cooling Limit
  ,                         !- Maximum Cooling Air Flow Rate (autosize)
  ,                         !- Maximum Total Cooling Capacity (autosize)
  ,                         !- Heating Availability Schedule
  ,                         !- Cooling Availability Schedule
  ,                         !- Dehumidification Control Type
  ,                         !- Cooling Sensible Heat Ratio
  ,                         !- Humidification Control Type
  ,                         !- Design Specification Outdoor Air Object Name
  ,                         !- Outdoor Air Inlet Node Name
  None;                     !- Demand Controlled Ventilation Type

ZoneHVAC:EquipmentConnections,
  {name},                   !- Zone Name
  {name}_EquipmentList,     !- Zone Conditioning Equipment List Name
  {name}_SupplyAirInlet,    !- Zone Air Inlet Node or NodeList Name
  {name}_ExhaustAirOutlet,  !- Zone Air Exhaust Node or NodeList Name
  {name}_ZoneAirNode,       !- Zone Air Node Name
  {name}_ZoneReturnAir;     !- Zone Return Air Node or NodeList Name

ZoneHVAC:EquipmentList,
  {name}_EquipmentList,
  SequentialLoad,
  ZoneHVAC:IdealLoadsAirSystem,
  {name}_IdealLoads,
  1, 1;
"""


def _people(z: ZoneSpec) -> str:
    if z.server_room:
        return ""
    sched = "OCC_MULTIPLIER"
    return f"""People,
  {z.name}_People,
  {z.name},
  {sched},                  !- Number of People Schedule
  People,
  {z.people},               !- Number of People
  ,                         !- People per Floor Area (blank)
  ,                         !- Floor Area per Person (blank)
  0.3,                      !- Fraction Radiant
  ,                         !- Sensible Heat Fraction (autocalculate default)
  ActivityLevel_Sched;
"""


def _lights(z: ZoneSpec) -> str:
    area = z.width * z.depth
    watts = area * z.lighting_wm2
    sched = "Always_On" if z.server_room else "OCC_MULTIPLIER"
    return f"""Lights,
  {z.name}_Lights,
  {z.name},
  {sched},
  LightingLevel,
  {watts},
  ,                         !- Watts per Floor Area (blank)
  ,                         !- Watts per Person (blank)
  ,                         !- Return Air Fraction (default 0)
  0.0,                      !- Fraction Radiant
  0.0,                      !- Fraction Visible
  1.0,                      !- Fraction Replaceable
  GeneralLights;
"""


def _equipment(z: ZoneSpec) -> str:
    area = z.width * z.depth
    watts = area * z.equip_wm2
    sched = "Always_On" if z.server_room else "OCC_MULTIPLIER"
    return f"""ElectricEquipment,
  {z.name}_Equip,
  {z.name},
  {sched},
  EquipmentLevel,
  {watts},
  ,                         !- Watts per Floor Area (blank)
  ,                         !- Watts per Person (blank)
  0.0,                      !- Fraction Latent
  0.5,                      !- Fraction Radiant
  0.0;                      !- Fraction Lost
"""


def _thermostat(z: ZoneSpec) -> str:
    return f"""ZoneControl:Thermostat,
  {z.name}_Thermostat,
  {z.name},
  Dual Zone Control Type Sched,
  ThermostatSetpoint:DualSetpoint,
  {z.name}_DualSP;

ThermostatSetpoint:DualSetpoint,
  {z.name}_DualSP,
  Heating_SP_Sched,         !- Heating Setpoint Temperature Schedule
  Cooling_SP_Sched;         !- Cooling Setpoint Temperature Schedule
"""


def _output_vars(z: ZoneSpec) -> str:
    return f"""Output:Variable,{z.name},Zone Air Temperature,TimeStep;
Output:Variable,{z.name},Zone Ideal Loads Supply Air Total Cooling Energy,TimeStep;
Output:Variable,{z.name},Zone Ideal Loads Supply Air Total Heating Energy,TimeStep;
"""


def generate_idf() -> str:
    header = f"""Version, {ENERGYPLUS_VERSION};

Building,
  BuilMirai Office,
  0.0,           !- North Axis {{deg}}
  City,          !- Terrain
  0.04,          !- Loads Convergence Tolerance
  0.4,           !- Temperature Convergence Tolerance
  FullInteriorAndExterior,
  5,             !- Maximum Number of Warmup Days
  1;             !- Minimum Number of Warmup Days

SimulationControl,
  No, No, No, No, Yes, No, 1;

Timestep, 4;   !- 4 per hour = 15 min intervals

RunPeriod,
  RunPeriod1, 7, 1, , 7, 1, , Monday, No, No, Yes, No, No;

Site:Location,
  London/Heathrow, 51.48, -0.45, 0.0, 24.0;

GlobalGeometryRules,
  UpperLeftCorner, CounterClockWise, World;

!- ===== Schedule Type Limits =====

ScheduleTypeLimits, Fraction, 0.0, 1.0, CONTINUOUS;
ScheduleTypeLimits, Temperature, -60, 200, CONTINUOUS;
ScheduleTypeLimits, Any Number;
ScheduleTypeLimits, Control Type, 0, 4, DISCRETE;

!- ===== Schedules =====

Schedule:Compact,
  Always_On, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 1.0;

Schedule:Compact,
  Always_Off, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 0.0;

Schedule:Compact,
  OCC_MULTIPLIER, Fraction,
  Through: 12/31, For: AllDays, Until: 24:00, 1.0;

Schedule:Compact,
  Office_Occ, Fraction,
  Through: 12/31,
  For: Weekdays,
    Until: 08:00, 0.0,
    Until: 18:00, 1.0,
    Until: 24:00, 0.0,
  For: AllOtherDays,
    Until: 24:00, 0.0;

Schedule:Compact,
  ActivityLevel_Sched, Any Number,
  Through: 12/31, For: AllDays, Until: 24:00, 120;

Schedule:Compact,
  Dual Zone Control Type Sched, Control Type,
  Through: 12/31, For: AllDays, Until: 24:00, 4;

Schedule:Compact,
  Heating_SP_Sched, Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 20.0;

Schedule:Compact,
  Cooling_SP_Sched, Temperature,
  Through: 12/31, For: AllDays, Until: 24:00, 26.0;

!- ===== Materials and Constructions =====

Material:NoMass,
  Adiabatic_Mat, Smooth, 100.0;

Material:NoMass,
  ExtWall_Mat, Smooth, 0.5;

Material:NoMass,
  Floor_Mat, Smooth, 0.25;

Construction, Adiabatic_Const, Adiabatic_Mat;
Construction, ExtWall_Const, ExtWall_Mat;
Construction, Floor_Const, Floor_Mat;

WindowMaterial:SimpleGlazingSystem,
  SimpleGlazing, 3.0, 0.3;

Construction, Window_Const, SimpleGlazing;
"""

    zones_idf = ""
    for z in ZONES:
        x1, y1 = z.x0, 0.0
        x2, y2 = z.x0 + z.width, z.depth
        zone_name = z.name

        zones_idf += f"\n!- ===== Zone: {zone_name} =====\n\n"
        zones_idf += f"Zone,\n  {zone_name},\n  0, 0, 0, 0,\n  1,             !- Multiplier\n  ,\n  {z.width * z.depth * H},  !- Volume {{m3}}\n  {z.width * z.depth};  !- Floor Area {{m2}}\n\n"

        # Surfaces
        zones_idf += _wall(f"{zone_name}_SouthWall", zone_name, x1, y1, x2, y2, 0, H, "S", "Outdoors")
        zones_idf += _wall(f"{zone_name}_NorthWall",  zone_name, x1, y1, x2, y2, 0, H, "N")
        zones_idf += _wall(f"{zone_name}_EastWall",   zone_name, x1, y1, x2, y2, 0, H, "E")
        zones_idf += _wall(f"{zone_name}_WestWall",   zone_name, x1, y1, x2, y2, 0, H, "W")
        zones_idf += _wall(f"{zone_name}_Floor",      zone_name, x1, y1, x2, y2, 0, H, "Floor", "Ground")
        zones_idf += _wall(f"{zone_name}_Ceiling",    zone_name, x1, y1, x2, y2, 0, H, "Ceiling")
        zones_idf += _window(f"{zone_name}_SouthWindow", f"{zone_name}_SouthWall", zone_name, x1, x2, y1, 0, H)

        # Internal gains
        zones_idf += _people(z)
        zones_idf += _lights(z)
        zones_idf += _equipment(z)

        # Thermostat
        zones_idf += _thermostat(z)

        # HVAC
        zones_idf += _zone_hvac(z)

        # Output variables
        zones_idf += _output_vars(z)

    footer = """
!- ===== Global Outputs =====

Output:Variable,*,Zone Air Temperature,TimeStep;
Output:Meter,Electricity:Facility,TimeStep;
OutputControl:Table:Style, Comma;
Output:Table:SummaryReports, AllSummary;
"""

    return header + zones_idf + footer


if __name__ == "__main__":
    content = generate_idf()
    OUTPUT_PATH.write_text(content)
    print(f"IDF written to {OUTPUT_PATH} ({len(content)} chars)")
