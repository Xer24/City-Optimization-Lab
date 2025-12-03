"""Global configuration for city-optimization-lab.
"""
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class GridConfig:
    width: int = 20
    height: int = 20
    spacing: float = 1.0
    diagonal: bool = False

    edge_keep: float = 0.9
    diag_keep: Optional[float] = None #optional -> can either be float or none
    population_range: Tuple[int, int] = (0,500)
    density_range: Tuple[float, float] = (0.01, 1)
    clusters_per_zone: int = 3

@dataclass
class TrafficConfig:
    trips_per_person: float = 0.3
    commercial_weight: float = 3.0
    industrial_weight: float = 2.0
    residential_weight: float = 1.0

@dataclass
class EnergyConfig:
    res_kwh_per_person: float = 5.0
    comm_kwh_per_person: float = 15.0
    ind_kwh_per_person: float = 25.0
    peak_hour: int = 10
    peak_multiplier: float = 1.7
    noise_std: float = 0.1

@dataclass
class SimulationConfig:

    seed: Optional[int] = 42

    grid: GridConfig = GridConfig()
    traffic: TrafficConfig = TrafficConfig()
    energy: EnergyConfig = EnergyConfig()

Sim_Config = SimulationConfig()