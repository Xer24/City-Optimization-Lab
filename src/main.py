import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# Make sure project root is importable when running `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel
from data.db import get_conn
from data.db import init_db, save_run  # <- your updated db.py


def compute_global_congestion(edge_flows) -> float:
    """
    A simple global congestion metric:
    mean edge flow across edges.
    Replace with a better formula later if you want.
    """
    if not edge_flows:
        return 0.0
    vals = list(edge_flows.values())
    return float(sum(vals) / len(vals))


def build_edge_id_map(city: CityGrid):
    """
    Create a stable mapping from each edge (u, v) to an integer edge_id.
    This is used to store edge_ticks.
    """
    edges = list(city.graph.edges())
    edge_to_id = {e: i for i, e in enumerate(edges)}
    return edge_to_id


def main():
    cfg = Sim_Config  # alias

    # ---- Create city ----
    city = CityGrid(
        width=cfg.grid.width,
        height=cfg.grid.height,
        spacing=cfg.grid.spacing,
        diagonal=cfg.grid.diagonal,
        seed=cfg.seed,
        edge_keep=cfg.grid.edge_keep,
        diag_keep=cfg.grid.diag_keep,
        population_range=cfg.grid.population_range,
        density_range=cfg.grid.density_range,
        clusters_per_zone=cfg.grid.clusters_per_zone,
    )

    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    # ---- Simulation ----
    sim = SimulationEngine(city)

    # ---- DB setup ----
    init_db()

    # Params you want stored with the run (edit freely)
    run_params = {
        "seed": cfg.seed,
        "grid": {
            "width": cfg.grid.width,
            "height": cfg.grid.height,
            "spacing": cfg.grid.spacing,
            "diagonal": cfg.grid.diagonal,
            "edge_keep": cfg.grid.edge_keep,
            "diag_keep": cfg.grid.diag_keep,
        },
        "traffic": {
            "trips_per_person": cfg.traffic.trips_per_person,
        },
    }

    sim_name = "city_sim"  # choose whatever label you want

    # Map graph edges -> edge_id for edge_ticks
    edge_to_id = build_edge_id_map(city)

    # Storage buffers (for this short run, storing in memory is fine)
    tick_rows = []  # [(tick, congestion, energy_usage), ...]
    edge_rows = []  # [(tick, edge_id, congestion), ...]

    # ---- Plot setup ----
    fig, (ax_city, ax_heat) = plt.subplots(1, 2, figsize=(12, 6))

    # ===== Tick 0 (initial state) =====
    traffic_model = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,
    )
    edge_flows = traffic_model.run_static_assignment()

    # City plot
    city.visualize(ax=ax_city, show=False, edge_flows=edge_flows)

    # Energy heatmap (tick 0)
    grid0 = sim.energy.daily_grid()
    im = ax_heat.imshow(grid0, origin="lower", aspect="equal")
    cbar = plt.colorbar(im, ax=ax_heat)
    cbar.set_label("Energy (sum over 24 hrs)")
    ax_heat.set_title("Tick 0")
    plt.tight_layout()
    plt.pause(0.1)

    # ---- Log tick 0 ----
    tick0 = 0
    congestion0 = compute_global_congestion(edge_flows)
    energy0 = float(grid0.sum())  # global energy usage for tick 0 (sum over grid)
    tick_rows.append((tick0, congestion0, energy0))

    for (u, v), flow in edge_flows.items():
        edge_id = edge_to_id.get((u, v))
        if edge_id is None:
            # Sometimes edges might be stored as (v, u) depending on your graph usage
            edge_id = edge_to_id.get((v, u))
        if edge_id is not None:
            edge_rows.append((tick0, int(edge_id), float(flow)))

    # ===== Animate over ticks =====
    num_ticks = 3  # change as needed

    for _ in range(num_ticks):
        # Advance simulation (updates tick internally)
        totals = sim.step()
        current_tick = int(sim.tick)

        # Recompute traffic
        traffic_model = TrafficModel(
            grid=city,
            trips_per_person=cfg.traffic.trips_per_person,
        )
        edge_flows = traffic_model.run_static_assignment()

        # Update city plot
        ax_city.clear()
        city.visualize(ax=ax_city, show=False, edge_flows=edge_flows)

        # Update heatmap + compute energy usage
        grid = sim.energy.daily_grid()
        im.set_data(grid)
        ax_heat.set_title(f"Tick {current_tick}")

        # Use grid sum as energy usage (consistent with tick 0)
        energy_usage = float(grid.sum())
        congestion = compute_global_congestion(edge_flows)

        print(f"Tick {current_tick}: energy_usage = {energy_usage:.2f}, congestion = {congestion:.4f}")

        # ---- Log this tick ----
        tick_rows.append((current_tick, congestion, energy_usage))

        for (u, v), flow in edge_flows.items():
            edge_id = edge_to_id.get((u, v))
            if edge_id is None:
                edge_id = edge_to_id.get((v, u))
            if edge_id is not None:
                edge_rows.append((current_tick, int(edge_id), float(flow)))

        plt.pause(0.7)

    # ---- Save to DB (auto run naming + summary) ----
    run_id = save_run(
        sim_name=sim_name,
        params=run_params,
        tick_rows=tick_rows,
        edge_rows=edge_rows
    )
    print(f"Saved run_id={run_id} with {len(tick_rows)} ticks and {len(edge_rows)} edge rows")
    with get_conn() as conn:
        df = pd.read_sql_query(
        "SELECT tick, congestion, energy_usage FROM ticks WHERE run_id=? ORDER BY tick",
        conn,
        params=(run_id,)
    )

    df.to_csv(f"data/run_{run_id}_ticks.csv", index=False)
    print(f"Wrote data/run_{run_id}_ticks.csv")
    plt.show()




if __name__ == "__main__":
    main()
