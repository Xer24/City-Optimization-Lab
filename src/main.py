"""
Main simulation runner with visualization and database logging.

Runs multi-modal urban transportation simulation with real-time
visualization of traffic flows and energy consumption. Saves results
to SQLite database and exports CSV files for analysis.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel
from data.db import get_conn, init_db, save_run
from simulation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


# ============================================================
# Metrics Computation
# ============================================================

def compute_global_congestion(edge_flows: Dict[Any, float]) -> float:
    """
    Compute mean edge flow across all edges.
    
    Args:
        edge_flows: Dictionary mapping edges to flow volumes
    
    Returns:
        Mean flow value
    """
    if not edge_flows:
        return 0.0
    vals = list(edge_flows.values())
    return float(sum(vals) / len(vals))


def max_flow(edge_flows: Dict[Any, float]) -> float:
    """
    Find maximum flow across all edges.
    
    Args:
        edge_flows: Dictionary mapping edges to flow volumes
    
    Returns:
        Maximum flow value
    """
    if not edge_flows:
        return 0.0
    return float(max(edge_flows.values()))


def total_flow(edge_flows: Dict[Any, float]) -> float:
    """
    Compute total flow across all edges.
    
    Args:
        edge_flows: Dictionary mapping edges to flow volumes
    
    Returns:
        Sum of all flows
    """
    if not edge_flows:
        return 0.0
    return float(sum(edge_flows.values()))


def compute_flow_statistics(
    flows_by_mode: Dict[str, Dict[Any, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive statistics for each mode.
    
    Args:
        flows_by_mode: Dictionary with keys "car", "ped", "transit"
    
    Returns:
        Nested dict with statistics per mode
    """
    stats = {}
    
    for mode, flows in flows_by_mode.items():
        if flows:
            vals = np.array(list(flows.values()))
            stats[mode] = {
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "total": float(np.sum(vals)),
                "p95": float(np.percentile(vals, 95)),
                "loaded_edges": int(np.sum(vals > 0)),
            }
        else:
            stats[mode] = {
                "mean": 0.0,
                "max": 0.0,
                "total": 0.0,
                "p95": 0.0,
                "loaded_edges": 0,
            }
    
    return stats


# ============================================================
# Database Helpers
# ============================================================

def build_edge_id_map(city: CityGrid) -> Dict[Tuple, int]:
    """
    Create stable mapping from edges to integer IDs.
    
    Args:
        city: CityGrid instance
    
    Returns:
        Dictionary mapping (u, v) tuples to edge IDs
    """
    edges = list(city.graph.edges())
    edge_map = {e: i for i, e in enumerate(edges)}
    logger.info(f"Created edge ID map with {len(edge_map)} edges")
    return edge_map


def prepare_tick_row(
    tick: int,
    flows: Dict[str, Dict[Any, float]],
    energy: float,
) -> Tuple:
    """
    Prepare database row for tick-level metrics.
    
    Args:
        tick: Tick number
        flows: Flows by mode
        energy: Energy consumption
    
    Returns:
        Tuple for database insertion
    """
    car_cong = compute_global_congestion(flows.get("car", {}))
    ped_cong = compute_global_congestion(flows.get("ped", {}))
    transit_cong = compute_global_congestion(flows.get("transit", {}))
    
    return (
        tick,
        round(car_cong, 4),
        round(ped_cong, 4),
        round(transit_cong, 4),
        round(energy, 4),
    )


def prepare_edge_mode_rows(
    tick: int,
    flows: Dict[str, Dict[Any, float]],
    edge_to_id: Dict[Tuple, int],
) -> List[Tuple]:
    """
    Prepare database rows for edge-mode flows.
    
    Args:
        tick: Tick number
        flows: Flows by mode
        edge_to_id: Edge ID mapping
    
    Returns:
        List of tuples for database insertion
    """
    rows = []
    
    for mode, mode_flows in flows.items():
        for (u, v), flow in mode_flows.items():
            edge_id = edge_to_id.get((u, v)) or edge_to_id.get((v, u))
            if edge_id is not None and flow > 0:
                rows.append((
                    tick,
                    int(edge_id),
                    mode,
                    round(float(flow), 4)
                ))
    
    return rows


# ============================================================
# Visualization
# ============================================================

def setup_visualization(
    figsize: Tuple[int, int] = (15, 7)
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Set up matplotlib figure with two subplots.
    
    Args:
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (figure, city_axis, heatmap_axis)
    """
    fig, (ax_city, ax_heat) = plt.subplots(1, 2, figsize=figsize)
    plt.ion()  # Interactive mode for live updates
    return fig, ax_city, ax_heat


def update_city_plot(
    ax: plt.Axes,
    city: CityGrid,
    tick: int,
    flows: Dict[str, Dict[Any, float]],
    vmax_by_mode: Dict[str, float],
) -> None:
    """
    Update city visualization with current flows.
    
    Args:
        ax: Matplotlib axis
        city: CityGrid instance
        tick: Current tick
        flows: Flows by mode
        vmax_by_mode: Maximum values for color scaling
    """
    ax.clear()
    city.visualize(
        ax=ax,
        show=False,
        flows_by_mode=flows,
        flow_vmax_by_mode=vmax_by_mode,
    )
    ax.set_title(f"Multi-modal Traffic Flows (Tick {tick})", fontweight="bold")


def update_energy_plot(
    ax: plt.Axes,
    tick: int,
    grid: np.ndarray,
    im: Optional[plt.cm.ScalarMappable] = None,
) -> plt.cm.ScalarMappable:
    """
    Update energy heatmap visualization.
    
    Args:
        ax: Matplotlib axis
        tick: Current tick
        grid: Energy grid data
        im: Existing image object (or None for first plot)
    
    Returns:
        Updated image object
    """
    if im is None:
        im = ax.imshow(grid, origin="lower", aspect="equal", cmap="hot")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Energy (kWh)", fontsize=10)
    else:
        im.set_data(grid)
    
    ax.set_title(f"Energy Consumption (Tick {tick})", fontweight="bold")
    return im


def print_tick_summary(
    tick: int,
    stats: Dict[str, Dict[str, float]],
    energy: float,
) -> None:
    """
    Print formatted summary of tick metrics.
    
    Args:
        tick: Tick number
        stats: Flow statistics by mode
        energy: Energy consumption
    """
    print(f"\n{'='*60}")
    print(f"TICK {tick}")
    print(f"{'='*60}")
    print(f"Energy: {energy:>12.2f} kWh")
    print(f"{'-'*60}")
    print(f"{'Mode':<12} {'Mean':>10} {'Max':>10} {'Total':>12} {'P95':>10}")
    print(f"{'-'*60}")
    
    for mode in ["car", "ped", "transit"]:
        if mode in stats:
            s = stats[mode]
            mode_name = mode.capitalize()
            print(
                f"{mode_name:<12} "
                f"{s['mean']:>10.2f} "
                f"{s['max']:>10.2f} "
                f"{s['total']:>12.0f} "
                f"{s['p95']:>10.2f}"
            )
    
    print(f"{'='*60}")


# ============================================================
# Main Simulation Loop
# ============================================================

def run_simulation(
    config: Any,
    num_ticks: int = 3,
    visualize: bool = True,
    save_db: bool = True,
    export_csv: bool = True,
    generate_report: bool = True,
    pause_duration: float = 0.7,
) -> int:
    """
    Run complete simulation with visualization and logging.
    
    Args:
        config: Configuration object
        num_ticks: Number of ticks to simulate
        visualize: Whether to show live visualization
        save_db: Whether to save to database
        export_csv: Whether to export CSV files
        generate_report: Whether to generate analysis report
        pause_duration: Seconds to pause between frames
    
    Returns:
        Database run_id (or 0 if not saved)
    
    Example:
        >>> from config import Sim_Config
        >>> run_id = run_simulation(Sim_Config, num_ticks=10, visualize=True)
    """
    cfg = config
    
    # ---- Create city ----
    logger.info("Creating city grid...")
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
    
    print("\n" + str(city))
    print(f"Nodes: {city.graph.number_of_nodes()}")
    print(f"Edges: {city.graph.number_of_edges()}")
    
    # ---- Initialize simulation ----
    logger.info("Initializing simulation engine...")
    sim = SimulationEngine(city)
    
    # ---- Initialize traffic model ----
    logger.info("Initializing traffic model...")
    traffic = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,
        rng_seed=cfg.seed,
    )
    
    print(f"\nModal shares: Car={traffic.car_share:.1%}, "
          f"Ped={traffic.ped_share:.1%}, Transit={traffic.transit_share:.1%}")
    
    # ---- Database setup ----
    if save_db:
        logger.info("Initializing database...")
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
        
        edge_to_id = build_edge_id_map(city)
    
    # Data collection
    tick_rows: List[Tuple] = []
    edge_mode_rows: List[Tuple] = []
    
    # ---- Visualization setup ----
    if visualize:
        fig, ax_city, ax_heat = setup_visualization()
        im_energy = None
        
        # Track max flows for consistent color scaling
        vmax_car = 0.0
        vmax_ped = 0.0
        vmax_transit = 0.0
    
    # ---- Run tick 0 ----
    logger.info("Running initial tick (Tick 0)...")
    tick0 = 0
    flows0 = traffic.run_multimodal_assignment(deterministic_demand=True)
    grid0 = sim.energy.daily_grid()
    energy0 = float(grid0.sum())
    
    stats0 = compute_flow_statistics(flows0)
    print_tick_summary(tick0, stats0, energy0)
    
    if save_db:
        tick_rows.append(prepare_tick_row(tick0, flows0, energy0))
        edge_mode_rows.extend(prepare_edge_mode_rows(tick0, flows0, edge_to_id))
    
    if visualize:
        # Update vmax values
        vmax_car = max(vmax_car, max_flow(flows0.get("car", {})))
        vmax_ped = max(vmax_ped, max_flow(flows0.get("ped", {})))
        vmax_transit = max(vmax_transit, max_flow(flows0.get("transit", {})))
        
        vmax_by_mode = {"car": vmax_car, "ped": vmax_ped, "transit": vmax_transit}
        
        # Initial plots
        update_city_plot(ax_city, city, tick0, flows0, vmax_by_mode)
        im_energy = update_energy_plot(ax_heat, tick0, grid0)
        
        plt.tight_layout()
        plt.pause(pause_duration)
    
    # ---- Main simulation loop ----
    logger.info(f"Running simulation for {num_ticks} ticks...")
    
    for i in range(num_ticks):
        # Advance simulation
        state = sim.step()
        current_tick = state.tick
        
        # Run traffic assignment
        flows = traffic.run_multimodal_assignment(deterministic_demand=True)
        
        # Get energy grid
        grid = sim.energy.daily_grid()
        energy = state.total_energy
        
        # Compute statistics
        stats = compute_flow_statistics(flows)
        print_tick_summary(current_tick, stats, energy)
        
        # Save data
        if save_db:
            tick_rows.append(prepare_tick_row(current_tick, flows, energy))
            edge_mode_rows.extend(prepare_edge_mode_rows(current_tick, flows, edge_to_id))
        
        # Update visualization
        if visualize:
            # Update vmax values
            vmax_car = max(vmax_car, max_flow(flows.get("car", {})))
            vmax_ped = max(vmax_ped, max_flow(flows.get("ped", {})))
            vmax_transit = max(vmax_transit, max_flow(flows.get("transit", {})))
            
            vmax_by_mode = {"car": vmax_car, "ped": vmax_ped, "transit": vmax_transit}
            
            # Update plots
            update_city_plot(ax_city, city, current_tick, flows, vmax_by_mode)
            im_energy = update_energy_plot(ax_heat, current_tick, grid, im_energy)
            
            fig.canvas.draw_idle()
            plt.pause(pause_duration)
    
    # ---- Save to database ----
    run_id = 0
    if save_db:
        logger.info("Saving to database...")
        run_id = save_run(
            sim_name="city_sim",
            params=run_params,
            tick_rows=tick_rows,
            edge_mode_rows=edge_mode_rows,
        )
        print(f"\nSaved run_id={run_id} with {len(tick_rows)} ticks "
              f"and {len(edge_mode_rows)} edge-mode records")
    
    # ---- Export CSV files ----
    if export_csv and save_db:
        logger.info("Exporting CSV files...")
        
        # Export tick-level data
        with get_conn() as conn:
            df_ticks = pd.read_sql_query(
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
        
        tick_csv_path = f"data/run_{run_id}_ticks.csv"
        df_ticks.to_csv(tick_csv_path, index=False)
        print(f"Exported: {tick_csv_path}")
        
        # Export edge-mode data
        with get_conn() as conn:
            df_edges = pd.read_sql_query(
                """
                SELECT tick, edge_id, mode, flow
                FROM edge_ticks_mode
                WHERE run_id=?
                ORDER BY tick, edge_id, mode
                """,
                conn,
                params=(run_id,),
            )
        
        edge_csv_path = f"data/run_{run_id}_edge_mode.csv"
        df_edges.to_csv(edge_csv_path, index=False)
        print(f"Exported: {edge_csv_path}")
        
        # ---- Generate Report ----
        if generate_report:
            logger.info("Generating simulation report...")
            try:
                reporter = ReportGenerator(output_dir="data/reports")
                reporter.add_tick_data(df_ticks)
                reporter.add_flow_data(df_edges)
                
                report_path = reporter.generate_report(
                    run_id=run_id,
                    run_params=run_params,
                    format="markdown",
                    include_plots=True,
                )
                print(f"Report generated: {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate report: {e}")
                print(f"Warning: Report generation failed: {e}")
    
    # ---- Show final plot ----
    if visualize:
        print("\nSimulation complete. Close plot window to exit.")
        plt.ioff()
        plt.show()
    
    return run_id


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-modal urban transportation simulation"
    )
    
    parser.add_argument(
        "--ticks",
        type=int,
        default=3,
        help="Number of simulation ticks (default: 3)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (faster)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database saving"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.7,
        help="Pause duration between frames (default: 0.7s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Override seed if provided
    if args.seed is not None:
        Sim_Config.seed = args.seed
    
    # Run simulation
    run_id = run_simulation(
        config=Sim_Config,
        num_ticks=args.ticks,
        visualize=not args.no_viz,
        save_db=not args.no_db,
        export_csv=not args.no_csv,
        generate_report=not args.no_report,
        pause_duration=args.pause,
    )
    
    logger.info(f"Simulation complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()