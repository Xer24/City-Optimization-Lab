"""Core simulation engine.

Coordinates traffic, energy, and other submodels over time.
"""

from models.traffic_model import TrafficModel
from models.energy_model import EnergyModel

class SimulationEngine:
    def __init__(self, city_grid, hours: int = 24):
        self.city_grid = city_grid
        self.hours = hours
        self.traffic_model = TrafficModel(city_grid)
        self.energy_model = EnergyModel(city_grid)

    def run(self):
        """Run a simple 24-hour stub simulation."""
        for t in range(self.hours):
            self.traffic_model.step()
            self.energy_model.step()
