import matplotlib.pyplot as plt
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel  


def main():
    # Create a 10x10 city grid
    city = CityGrid(
        width=10,
        height=10,
        spacing=1.0,
        diagonal=False,
        seed=44,
        edge_keep=0.9,
        diag_keep=None,
        population_range=(0, 500),
        density_range=(0.1, 1.0),
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
        trips_per_person=0.4,  # adjust to increase/decrease traffic
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
    num_ticks = 3
    for _ in range(num_ticks):
        # Advance simulation (this will usually update city / energy)
        totals = sim.step()

        # Recompute traffic for updated city state
        traffic_model = TrafficModel(
            grid=city,
            trips_per_person=0.4,
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
