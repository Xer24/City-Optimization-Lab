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
from data.db import get_conn, init_db, save_run


def compute_global_congestion(edge_flows) -> float:
    """Mean edge flow across edges."""
    if not edge_flows:
        return 0.0
    vals = list(edge_flows.values())
    return float(sum(vals) / len(vals))


def build_edge_id_map(city: CityGrid):
    """Create a stable mapping from each edge (u, v) to an integer edge_id."""
    edges = list(city.graph.edges())
    return {e: i for i, e in enumerate(edges)}


def max_flow(edge_flows) -> float:
    if not edge_flows:
        return 0.0
    return float(max(edge_flows.values()))


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

    # ---- Simulation (energy, tick clock, etc.) ----
    sim = SimulationEngine(city)

    # ---- Traffic model (multi-modal) ----
    traffic = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,
        rng_seed=cfg.seed,
    )

    # ---- DB setup ----
    init_db()

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
            "mode": "multimodal_overlay",
            "shares": {
                "car": traffic.car_share,
                "ped": traffic.ped_share,
                "transit": traffic.transit_share,
            },
        },
    }

    sim_name = "city_sim"

    edge_to_id = build_edge_id_map(city)

    # tick_rows: (tick, car_cong, ped_cong, transit_cong, energy_usage)
    tick_rows = []

    # edge_mode_rows: (tick, edge_id, mode, flow)
    edge_mode_rows = []

    # ---- Plot setup: overlay city + energy ----
    fig, (ax_city, ax_heat) = plt.subplots(1, 2, figsize=(15, 7))

    # Per-mode vmax for consistent color scaling over time
    vmax_car = 0.0
    vmax_ped = 0.0
    vmax_transit = 0.0

    # ===== Tick 0 =====
    tick0 = 0
    flows0 = traffic.run_multimodal_assignment(deterministic_demand=True)

    vmax_car = max(vmax_car, max_flow(flows0["car"]))
    vmax_ped = max(vmax_ped, max_flow(flows0["ped"]))
    vmax_transit = max(vmax_transit, max_flow(flows0["transit"]))

    ax_city.clear()
    city.visualize(
        ax=ax_city,
        show=False,
        flows_by_mode=flows0,
        flow_vmax_by_mode={"car": vmax_car, "ped": vmax_ped, "transit": vmax_transit},
    )
    ax_city.set_title("Multi-modal flows (Tick 0)")

    grid0 = sim.energy.daily_grid()
    im = ax_heat.imshow(grid0, origin="lower", aspect="equal")
    cbar = plt.colorbar(im, ax=ax_heat)
    cbar.set_label("Energy (sum over 24 hrs)")
    ax_heat.set_title("Energy (Tick 0)")

    plt.tight_layout()
    fig.canvas.draw_idle()
    plt.pause(0.1)

    # Log tick 0
    car_cong0 = compute_global_congestion(flows0["car"])
    ped_cong0 = compute_global_congestion(flows0["ped"])
    transit_cong0 = compute_global_congestion(flows0["transit"])
    energy0 = float(grid0.sum())

    tick_rows.append(
        (tick0, round(car_cong0, 4), round(ped_cong0, 4), round(transit_cong0, 4), round(energy0, 4))
    )

    for mode, mode_flows in flows0.items():
        for (u, v), flow in mode_flows.items():
            edge_id = edge_to_id.get((u, v)) or edge_to_id.get((v, u))
            if edge_id is not None:
                edge_mode_rows.append((tick0, int(edge_id), mode, round(float(flow), 4)))

    # ===== Animate over ticks =====
    num_ticks = 3

    for _ in range(num_ticks):
        sim.step()
        current_tick = int(sim.tick)

        flows = traffic.run_multimodal_assignment(deterministic_demand=True)

        vmax_car = max(vmax_car, max_flow(flows["car"]))
        vmax_ped = max(vmax_ped, max_flow(flows["ped"]))
        vmax_transit = max(vmax_transit, max_flow(flows["transit"]))

        ax_city.clear()
        city.visualize(
            ax=ax_city,
            show=False,
            flows_by_mode=flows,
            flow_vmax_by_mode={"car": vmax_car, "ped": vmax_ped, "transit": vmax_transit},
        )
        ax_city.set_title(f"Multi-modal flows (Tick {current_tick})")

        grid = sim.energy.daily_grid()
        im.set_data(grid)
        ax_heat.set_title(f"Energy (Tick {current_tick})")

        energy_usage = float(grid.sum())
        car_cong = compute_global_congestion(flows["car"])
        ped_cong = compute_global_congestion(flows["ped"])
        transit_cong = compute_global_congestion(flows["transit"])

        print(
            f"Tick {current_tick}: energy={energy_usage:.2f} | "
            f"car={car_cong:.4f}, ped={ped_cong:.4f}, transit={transit_cong:.4f}"
        )

        tick_rows.append(
            (current_tick, round(car_cong, 4), round(ped_cong, 4), round(transit_cong, 4), round(energy_usage, 4))
        )

        for mode, mode_flows in flows.items():
            for (u, v), flow in mode_flows.items():
                edge_id = edge_to_id.get((u, v)) or edge_to_id.get((v, u))
                if edge_id is not None:
                    edge_mode_rows.append((current_tick, int(edge_id), mode, round(float(flow), 4)))

        fig.canvas.draw_idle()
        plt.pause(0.7)

    # ---- Save to DB ----
    run_id = save_run(
        sim_name=sim_name,
        params=run_params,
        tick_rows=tick_rows,
        edge_mode_rows=edge_mode_rows,
    )
    print(f"Saved run_id={run_id} with {len(tick_rows)} ticks and {len(edge_mode_rows)} edge-mode rows")

    # ---- Export tick-level CSV ----
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                tick,
                car_congestion,
                ped_congestion,
                transit_congestion,
                energy_usage
            FROM ticks
            WHERE run_id=?
            ORDER BY tick
            """,
            conn,
            params=(run_id,),
        )
    df = df.round(4)
    df.to_csv(f"data/run_{run_id}_ticks.csv", index=False)
    print(f"Wrote data/run_{run_id}_ticks.csv")

    # ---- Export edge-mode CSV ----
    with get_conn() as conn:
        df2 = pd.read_sql_query(
            """
            SELECT tick, edge_id, mode, flow
            FROM edge_ticks_mode
            WHERE run_id=?
            ORDER BY tick, edge_id, mode
            """,
            conn,
            params=(run_id,),
        )
    df2 = df2.round(4)
    df2.to_csv(f"data/run_{run_id}_edge_mode.csv", index=False)
    print(f"Wrote data/run_{run_id}_edge_mode.csv")

    plt.show()


if __name__ == "__main__":
    main()
