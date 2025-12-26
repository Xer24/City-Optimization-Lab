"""Core simulation engine.

Coordinates energy and other submodels over time.
Traffic is handled in main.py (multi-modal).
"""
from __future__ import annotations

from models.city_grid import CityGrid
from models.energy_model import EnergyModel
from config import Sim_Config as cfg


class SimulationEngine:
    def __init__(self, city: CityGrid):
        self.city = city
        self.energy = EnergyModel(
            city=city,
            noise_std=cfg.energy.noise_std,
            rng_seed=cfg.seed + cfg.energy.rng_offset,
        )
        self.tick = 0

    def step(self):
        """
        Run one tick (= 24hrs).
        Updates energy + increments tick.
        Returns energy totals if you want to use it later.
        """
        total_demand_by_hour = self.energy.simulate_day(range(24))
        self.tick += 1
        return {"total_demand_by_hour": total_demand_by_hour}
