"""Core simulation engine.

Coordinates traffic, energy, and other submodels over time.
"""

from models.traffic_model import TrafficModel
from models.city_grid import CityGrid
from models.energy_model import EnergyModel

class SimulationEngine:
    # Controls time progression (ticks)
    
    def __init__(self, city: CityGrid):
        self.city = city
        #self.traffic_model = TrafficModel(city_grid)
        self.energy = EnergyModel(city)
        self.tick = 0
    
    def step(self, *, plot: bool = False):
        #run one tick = 24hrs of the simulation
        total_demand_by_hour = self.energy.simulate_day(range(24)) #compute 24hr demand
        heatmap_grid = self.energy.plot_heatmap(show = plot) #compute spatial distribution

        #Later: evolve city for next tick (population, zoning etc etc)

        self.tick += 1

        return total_demand_by_hour, heatmap_grid
