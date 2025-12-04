"""Energy demand and supply model."""
# Energy equation being used is E(t) = baseline * zone(t) (1+alpha*population/avg pop) * (1 + beta*denisty)
    #E(t): Energy demand
    #baseline: A baseline energy -> like a base power requirement 
    # (1+alpha*population/avg pop): Normalize by mean population 
    #Zone: different energy demand based on zone
    #(1 + beta*denisty): takes into account density of buildings

from __future__ import annotations
from dataclasses import dataclass
from models.city_grid import CityGrid
from typing import Dict, Hashable, Optional, Iterable


import numpy as np
import random
import matplotlib.pyplot as plt
Node = Hashable #mean any object that is hashable

@dataclass #only used to hold data
class DemandProfile:
    # represents a 24-hr demand curve
    name: str
    values: np.ndarray

    def energy_factor(self, hour: int) -> float:
        return float(self.values[int(hour) % 24])
    
def default_profiles() -> Dict[str, DemandProfile]: # construct demand multiplier curve for each zone and puts them in a dict
    # so we are setting the values for the multipliers at these ranges of hours
    hours = np.arange(24)
    residential = np.where(
        (6 <= hours) & (hours < 9),
        1.3,
        np.where((15 <= hours) & (hours < 22), 1.6,0.8),
        )
    commercial = np.where(
        (9 <= hours) & (hours < 18),
        1.7,
        0.4,
    )
    industrial = np.where (
        (7 <= hours) & (hours < 19),
        1.2,
        0.9,
    )

    return {
        "residential":DemandProfile("residential", residential),
        "commercial": DemandProfile("commercial", commercial),
        "industrial": DemandProfile("industrial", industrial),
    }

class EnergyModel:
    #computer hourly energy demand for each city node
    # each nodes needs zoning, baseline_energy, population and density
    def __init__(self, 
    city: "CityGrid",
    *,
    profiles: Optional[Dict[str, DemandProfile]] = None, #optional parameter
    population_scale: float = 0.5,
    density_scale: float = 0.8,
    noise_std: float = 0.1,
    rng_seed: int | None = None
    ):
        self.city = city
        self.graph = city.graph
        self.population_scale = float(population_scale)
        self.density_scale = float(density_scale)
        self.profiles = profiles if profiles is not None else default_profiles()
        pops = [float(data.get("population",0.0))
        for _, data in self.graph.nodes(data = True)] #compuate avg pop for scaling
        self.avg_pop = np.mean(pops) if len(pops) > 0 else 1.0
        self.noise_std = float(noise_std)
        self.rng = random.Random(rng_seed)


    
    def node_demand(self, node: Node, hour: int) -> float:
        #computer energy demand for a single node at a given hour
        data = self.graph.nodes[node]
        baseline = float(data.get("baseline_energy", 0.0))
        zoning = str(data.get("zoning", "residential")).lower()
        population = float(data.get("population", 0.0))
        density = float(data.get("density", 0.0))

        profile = self.profiles.get(zoning) # get zone
        zone_factor = profile.energy_factor(hour) if profile else 1.0 # the zone(t) term

        if self.avg_pop > 0: #creates the normalized population term
            pop_factor = 1.0 + self.population_scale * (population/ self.avg_pop)
        else:
            pop_factor = 1.0 
        density_factor = 1.0 + self.density_scale * density #Density term
        
        demand = baseline * zone_factor * pop_factor * density_factor
        noise = self.rng.gauss(1.0, self.noise_std)

        return demand * max(noise, 0.1)  #randomized energy equation per hour
    
    def city_demand(self, hour: int) -> float:
        #city demand per hour
        return float(sum(self.node_demand(node,hour) for node in self.graph.nodes))
    
    def simulate_day(self, hours: Iterable[int] = range(24)) -> np.ndarray:
        #total city demand over a day
        hours = list(hours)
        demand = np.empty(len(hours), dtype = float)
        for i, h in enumerate(hours):
            demand[i] = self.city_demand(h)
        return demand
    def simulate_day_by_node(
        self,
        hours: Iterable[int] = range(24),) -> Dict[Node, np.ndarray]:
        # simulate per node energy demand over hours

        hours = list(hours)
        result: Dict[Node, np.ndarray] = {}
        
        for node in self.graph.nodes:
            arr = np.zeros(len(hours), dtype = float)
            for i, h in enumerate(hours):
                arr[i] = self.node_demand(node, h)
            result[node] = arr
        return result 
    def plot_heatmap(
        self,
        *,
        ax = None,
        show: bool = True) -> np.ndarray:
        # plot a heatmap of total energy demand per node over 1 tick (24hrs)
        height = getattr(self.city, "height",None)
        width = getattr(self.city, "width", None)
        if height is None or width is None:
            raise ValueError("City Grid must have a height and width")
        
        grid = np.zeros((height, width), dtype = float)
        for node, data in self.graph.nodes(data = True):
            row = data.get("row")
            col = data.get("col")
            if row is None or col is None:
                continue
            total = 0.0
            for hour in range(24):
                total += self.node_demand(node, hour)
            grid[row, col] = total
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        im = ax.imshow(grid, origin = "lower", aspect = "equal")
        ax.set_title("Total 24-Hour Energy Demand (One Tick)")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Energy (sum over 24 hrs)")

        if created_fig and show:
            plt.show()

        return grid
# Helper
    def daily_grid(self) -> np.ndarray:
        #return grid of total energy per node over 24 hours
        height = getattr(self.city, "height", None)
        width = getattr(self.city, "width", None)
        if height is None or width is None:
            raise ValueError("City Grid must have a height and width")
        
        grid = np.zeros((height,width), dtype = float)

        for node, data in self.graph.nodes(data = True):
            row = data.get("row")
            col = data.get("col")
            if row is None or col is None:
                continue
            total = 0.0
            for hour in range(24):
                total += self.node_demand(node, hour)
            grid[row, col] = total
        return grid 

