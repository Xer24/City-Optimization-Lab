"""Entry point for running simulations from the command line.

Example:
    python src/main.py
"""

from models.city_grid import CityGrid
from simulation.engine import SimulationEngine

def main():
    # TODO: replace this stub with real wiring to configs and CLI args
    city = CityGrid()
    sim = SimulationEngine(city)
    sim.run()
    print("Simulation run complete (stub).")

if __name__ == "__main__":
    main()
