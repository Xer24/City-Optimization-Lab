import matplotlib.pyplot as plt

from config import Sim_Config          # 🔹 NEW: central config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel  


def main():
    cfg = Sim_Config  # shorter alias

    # Create a city grid using values from the config
    city = CityGrid(
        width=cfg.grid.width,
        height=cfg.grid.height,
        spacing=cfg.grid.spacing,
        diagonal=cfg.grid.diagonal,
        seed=cfg.seed,                          # global RNG seed
        edge_keep=cfg.grid.edge_keep,
        diag_keep=cfg.grid.diag_keep,
        population_range=cfg.grid.population_range,
        density_range=cfg.grid.density_range,
        clusters_per_zone=cfg.grid.clusters_per_zone,
    )

    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    # Set up simulation (energy, etc.)
    sim = SimulationEngine(city)

    # Figure with two subplots: left = city (with traffic), right = heatmap
    fig, (ax_city, ax_heat) = plt.subplots(1, 2, figsize=(12, 6))

    # ----- Tick 0: initial traffic + energy -----

    # Initial traffic based on current city state
    traffic_model = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,  # from config instead of 0.4
        # If you want to wire weights as well, uncomment and make sure names match:
        # commerical_weight=cfg.traffic.commercial_weight,
        # industrail_weight=cfg.traffic.industrial_weight,
        # residential_weight=cfg.traffic.residential_weight,
    )
    edge_flows = traffic_model.run_static_assignment()

    # Draw city with traffic for tick 0
    city.visualize(ax=ax_city, show=False, edge_flows=edge_flows)

    # Initial energy heatmap grid (for tick 0)
    initial_grid = sim.energy.daily_grid()
    im = ax_heat.imshow(initial_grid, origin="lower", aspect="equal")
    cbar = plt.colorbar(im, ax=ax_heat)
    cbar.set_label("Energy (sum over 24 hrs)")
    ax_heat.set_title("Tick 0")

    plt.tight_layout()
    plt.pause(0.1)  # initial draw

    # ----- Animate over ticks -----
    num_ticks = 3  # you *could* also move this into SIM_CONFIG if you want later

    for _ in range(num_ticks):
        # Advance simulation (this will usually update city / energy)
        totals = sim.step()

        # Recompute traffic for updated city state
        traffic_model = TrafficModel(
            grid=city,
            trips_per_person=cfg.traffic.trips_per_person,
            # same comment as above if you want weights:
            # commerical_weight=cfg.traffic.commercial_weight,
            # industrail_weight=cfg.traffic.industrial_weight,
            # residential_weight=cfg.traffic.residential_weight,
        )
        edge_flows = traffic_model.run_static_assignment()

        # Update city plot with new traffic
        ax_city.clear()
        city.visualize(ax=ax_city, show=False, edge_flows=edge_flows)

        # Update heatmap
        grid = sim.energy.daily_grid()
        im.set_data(grid)
        ax_heat.set_title(f"Tick {sim.tick}")
        print(f"Tick {sim.tick}: total daily demand = {totals.sum():.2f}")

        plt.pause(0.7)  # small pause so you can see each frame

    # keep window open at the end
    plt.show()


if __name__ == "__main__":
    main()
