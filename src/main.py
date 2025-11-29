import matplotlib.pyplot as plt
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

    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    # Set up simulation
    sim = SimulationEngine(city)

    # Create figure with two subplots: left = city, right = heatmap
    fig, (ax_city, ax_heat) = plt.subplots(1, 2, figsize=(12, 6))

    # Draw static city layout on the left
    city.visualize(ax=ax_city, show=False)

    # Initial heatmap grid (for tick 0)
    initial_grid = sim.energy.daily_grid()
    im = ax_heat.imshow(initial_grid, origin="lower", aspect="equal")
    cbar = plt.colorbar(im, ax=ax_heat)
    cbar.set_label("Energy (sum over 24 hrs)")
    ax_heat.set_title("Tick 0")

    plt.tight_layout()
    plt.pause(0.1)  # draw once

    # Animate over ticks
    num_ticks = 3
    for _ in range(num_ticks):
        totals = sim.step()  # advances tick, recomputes 24h demand

        # recompute daily grid based on updated state
        grid = sim.energy.daily_grid()
        im.set_data(grid)  # update heatmap image

        ax_heat.set_title(f"Tick {sim.tick}")
        print(f"Tick {sim.tick}: total daily demand = {totals.sum():.2f}")

        plt.pause(0.7)  # pause so you can see it update

    # keep window open at the end
    plt.show()


if __name__ == "__main__":
    main()
