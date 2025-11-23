from models.city_grid import CityGrid
from simulation.engine import SimulationEngine


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

    # Basic debug info
    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    some_node = (0, 0)
    print("Node (0,0) attrs:", city.graph.nodes[some_node])

    some_edge = next(iter(city.graph.edges))
    print("Edge", some_edge, "attrs:", city.graph.edges[some_edge])

    # Optional: visualize the city layout once at the start
    city.visualize()

    # Set up the simulation engine (handles ticks + energy model)
    sim = SimulationEngine(city)

    # Run a few ticks (each tick = one 24-hour cycle)
    num_ticks = 3
    for _ in range(num_ticks):
        total_demand_by_hour, heatmap_grid = sim.step(plot=True)

        print(f"\n=== Tick {sim.tick} ===")
        print("Total city demand by hour:", total_demand_by_hour)
        print("Total daily demand (sum over 24h):",
              total_demand_by_hour.sum())


if __name__ == "__main__":
    main()
