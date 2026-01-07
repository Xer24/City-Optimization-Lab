"""
Policy evaluation and comparison tool.

Evaluates baseline and optimized policies over longer simulation horizons
to assess real-world performance. Provides detailed metrics and visualization
support for policy analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel

logger = logging.getLogger(__name__)


# ============================================================
# Policy and Metrics
# ============================================================

@dataclass
class Policy:
    """
    Urban transportation policy parameters.
    
    Attributes:
        transit_frequency_mult: Transit service frequency multiplier
        congestion_toll: Road pricing in $/trip
        road_capacity_scale: Road capacity scaling factor
    """
    transit_frequency_mult: float = 1.0
    congestion_toll: float = 0.0
    road_capacity_scale: float = 1.0


@dataclass
class TickMetrics:
    """
    Per-tick simulation metrics.
    
    Attributes:
        tick: Simulation tick number
        energy: Energy consumption
        car_mean: Mean car flow
        ped_mean: Mean pedestrian flow
        transit_mean: Mean transit flow
        mean_cong: Mean congestion across modes
        p95_cong: 95th percentile congestion
        travel_time_proxy: Travel time proxy
    """
    tick: int
    energy: float
    car_mean: float
    ped_mean: float
    transit_mean: float
    mean_cong: float
    p95_cong: float
    travel_time_proxy: float


@dataclass
class SummaryMetrics:
    """
    Summary metrics aggregated over simulation.
    
    Attributes:
        avg_travel_time: Average travel time
        avg_energy: Average energy consumption
        mean_congestion: Mean congestion
        congestion_p95: 95th percentile congestion
        total_car_trips: Total car trips
        total_ped_trips: Total pedestrian trips
        total_transit_trips: Total transit trips
    """
    avg_travel_time: float
    avg_energy: float
    mean_congestion: float
    congestion_p95: float
    total_car_trips: float = 0.0
    total_ped_trips: float = 0.0
    total_transit_trips: float = 0.0


# ============================================================
# Simulation Helpers
# ============================================================

def mean_flow(edge_flows: Dict[Any, float]) -> float:
    """Calculate mean flow across edges."""
    if not edge_flows:
        return 0.0
    vals = np.fromiter(edge_flows.values(), dtype=float)
    return float(vals.mean()) if vals.size > 0 else 0.0


def p95_flow(edge_flows: Dict[Any, float]) -> float:
    """Calculate 95th percentile flow."""
    if not edge_flows:
        return 0.0
    vals = np.fromiter(edge_flows.values(), dtype=float)
    return float(np.percentile(vals, 95)) if vals.size > 0 else 0.0


def total_flow(edge_flows: Dict[Any, float]) -> float:
    """Calculate total flow across all edges."""
    if not edge_flows:
        return 0.0
    return float(sum(edge_flows.values()))


def apply_capacity_scale(city: CityGrid, scale: float) -> None:
    """
    Scale road capacities by factor.
    
    Args:
        city: CityGrid instance
        scale: Capacity scaling factor
    """
    for u, v, data in city.graph.edges(data=True):
        if "capacity" in data and data["capacity"] is not None:
            data["capacity"] = float(data["capacity"]) * float(scale)


def apply_policy_to_mode_shares(
    traffic: TrafficModel,
    policy: Policy,
) -> None:
    """
    Convert policy parameters to modal shares.
    
    Args:
        traffic: TrafficModel instance
        policy: Policy to apply
    """
    # Start from current shares
    car = float(traffic.car_share)
    ped = float(traffic.ped_share)
    transit = float(traffic.transit_share)
    
    # Policy effects
    car_shift = 0.05 * (policy.congestion_toll / 5.0)
    transit_shift = 0.08 * (policy.transit_frequency_mult - 1.0)
    
    # Apply shifts
    car = car * (1.0 - car_shift)
    transit = transit * (1.0 + max(-0.3, min(0.5, transit_shift)))
    
    # Clip and normalize
    car = max(0.05, min(0.9, car))
    transit = max(0.05, min(0.9, transit))
    ped = max(0.05, 1.0 - car - transit)
    
    total = car + transit + ped
    traffic.car_share = float(car / total)
    traffic.transit_share = float(transit / total)
    traffic.ped_share = float(ped / total)


# ============================================================
# Core Evaluation
# ============================================================

def run_episode(
    policy: Policy,
    seed: int,
    ticks: int,
) -> Tuple[SummaryMetrics, List[TickMetrics]]:
    """
    Run simulation episode with given policy.
    
    Args:
        policy: Policy to evaluate
        seed: Random seed
        ticks: Number of simulation ticks
    
    Returns:
        Tuple of (summary_metrics, per_tick_metrics)
    
    Example:
        >>> policy = Policy(transit_frequency_mult=1.3)
        >>> summary, ticks = run_episode(policy, seed=42, ticks=20)
        >>> print(f"Avg travel time: {summary.avg_travel_time:.2f}")
    """
    cfg = Sim_Config
    
    # Create city
    city = CityGrid(
        width=cfg.grid.width,
        height=cfg.grid.height,
        spacing=cfg.grid.spacing,
        diagonal=cfg.grid.diagonal,
        seed=seed,
        edge_keep=cfg.grid.edge_keep,
        diag_keep=cfg.grid.diag_keep,
        population_range=cfg.grid.population_range,
        density_range=cfg.grid.density_range,
        clusters_per_zone=cfg.grid.clusters_per_zone,
    )
    
    # Apply policy
    apply_capacity_scale(city, policy.road_capacity_scale)
    
    # Initialize simulation
    sim = SimulationEngine(city)
    
    # Initialize traffic model
    traffic = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,
        rng_seed=seed,
    )
    apply_policy_to_mode_shares(traffic, policy)
    
    # Collect per-tick metrics
    tick_rows: List[TickMetrics] = []
    total_car = 0.0
    total_ped = 0.0
    total_transit = 0.0
    
    for _ in range(ticks):
        current_tick = int(sim.tick)
        
        # Run traffic assignment
        flows = traffic.run_multimodal_assignment(deterministic_demand=True)
        
        # Flow metrics per mode
        car_mean = mean_flow(flows.get("car", {}))
        ped_mean = mean_flow(flows.get("ped", {}))
        transit_mean = mean_flow(flows.get("transit", {}))
        
        # Accumulate total trips
        total_car += total_flow(flows.get("car", {}))
        total_ped += total_flow(flows.get("ped", {}))
        total_transit += total_flow(flows.get("transit", {}))
        
        # Congestion metrics
        mean_cong = (car_mean + ped_mean + transit_mean) / 3.0
        p95_cong = max(
            p95_flow(flows.get("car", {})),
            p95_flow(flows.get("ped", {})),
            p95_flow(flows.get("transit", {})),
        )
        
        # Travel time proxy
        travel_time_proxy = 1.2 * car_mean + 0.8 * transit_mean + 0.6 * ped_mean
        
        # Energy consumption
        grid = sim.energy.daily_grid()
        energy = float(grid.sum())
        
        # Store metrics
        tick_rows.append(
            TickMetrics(
                tick=current_tick,
                energy=energy,
                car_mean=car_mean,
                ped_mean=ped_mean,
                transit_mean=transit_mean,
                mean_cong=mean_cong,
                p95_cong=p95_cong,
                travel_time_proxy=travel_time_proxy,
            )
        )
        
        # Advance simulation
        sim.step()
    
    # Compute summary metrics
    summary = SummaryMetrics(
        avg_travel_time=float(np.mean([r.travel_time_proxy for r in tick_rows])),
        avg_energy=float(np.mean([r.energy for r in tick_rows])),
        mean_congestion=float(np.mean([r.mean_cong for r in tick_rows])),
        congestion_p95=float(np.max([r.p95_cong for r in tick_rows])),
        total_car_trips=float(total_car),
        total_ped_trips=float(total_ped),
        total_transit_trips=float(total_transit),
    )
    
    return summary, tick_rows


# ============================================================
# I/O and Comparison
# ============================================================

def load_best_policy(best_json_path: str) -> Policy:
    """
    Load best policy from JSON file.
    
    Args:
        best_json_path: Path to JSON file with best policy
    
    Returns:
        Policy instance
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    if not os.path.exists(best_json_path):
        raise FileNotFoundError(f"Policy file not found: {best_json_path}")
    
    with open(best_json_path, "r") as f:
        data = json.load(f)
    
    # Handle both direct policy JSON and nested format
    policy_data = data.get("best_policy", data)
    
    return Policy(
        transit_frequency_mult=float(policy_data.get("transit_frequency_mult", 1.0)),
        congestion_toll=float(policy_data.get("congestion_toll", 0.0)),
        road_capacity_scale=float(policy_data.get("road_capacity_scale", 1.0)),
    )


def save_tick_csv(path: str, rows: List[TickMetrics]) -> None:
    """
    Save per-tick metrics to CSV file.
    
    Args:
        path: Output CSV file path
        rows: List of TickMetrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(rows)} tick metrics to {path}")


def save_comparison_json(
    path: str,
    baseline_policy: Policy,
    best_policy: Policy,
    baseline_summary: SummaryMetrics,
    best_summary: SummaryMetrics,
) -> None:
    """
    Save policy comparison to JSON file.
    
    Args:
        path: Output JSON file path
        baseline_policy: Baseline policy
        best_policy: Optimized policy
        baseline_summary: Baseline metrics
        best_summary: Optimized metrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Compute improvements
    improvements = {}
    for metric in ["avg_travel_time", "avg_energy", "mean_congestion", "congestion_p95"]:
        baseline_val = getattr(baseline_summary, metric)
        best_val = getattr(best_summary, metric)
        
        if baseline_val > 0:
            pct_change = 100.0 * (best_val - baseline_val) / baseline_val
        else:
            pct_change = 0.0
        
        improvements[metric] = {
            "baseline": baseline_val,
            "optimized": best_val,
            "absolute_change": best_val - baseline_val,
            "percent_change": pct_change,
        }
    
    payload = {
        "baseline_policy": asdict(baseline_policy),
        "optimized_policy": asdict(best_policy),
        "baseline_metrics": asdict(baseline_summary),
        "optimized_metrics": asdict(best_summary),
        "improvements": improvements,
    }
    
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"Saved comparison to {path}")


def print_comparison(
    baseline: SummaryMetrics,
    best: SummaryMetrics,
) -> None:
    """
    Print formatted comparison table.
    
    Args:
        baseline: Baseline metrics
        best: Optimized metrics
    """
    def fmt(x: float) -> str:
        return f"{x:.4f}"
    
    def pct(x: float, base: float) -> str:
        if base == 0:
            return "N/A"
        return f"{100.0 * (x - base) / base:+.2f}%"
    
    print("\n" + "=" * 80)
    print("POLICY EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ("avg_travel_time", "Avg Travel Time"),
        ("avg_energy", "Avg Energy"),
        ("mean_congestion", "Mean Congestion"),
        ("congestion_p95", "P95 Congestion"),
        ("total_car_trips", "Total Car Trips"),
        ("total_ped_trips", "Total Ped Trips"),
        ("total_transit_trips", "Total Transit Trips"),
    ]
    
    for attr, label in metrics_to_compare:
        baseline_val = getattr(baseline, attr)
        best_val = getattr(best, attr)
        change = pct(best_val, baseline_val)
        
        print(f"{label:<25} {fmt(baseline_val):>15} {fmt(best_val):>15} {change:>15}")
    
    print("=" * 80)
    
    # Mode share comparison
    baseline_total = baseline.total_car_trips + baseline.total_ped_trips + baseline.total_transit_trips
    best_total = best.total_car_trips + best.total_ped_trips + best.total_transit_trips
    
    if baseline_total > 0 and best_total > 0:
        print("\nMODAL SHARES")
        print("-" * 80)
        print(f"{'Mode':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
        print("-" * 80)
        
        for mode, base_val, best_val in [
            ("Car", baseline.total_car_trips, best.total_car_trips),
            ("Pedestrian", baseline.total_ped_trips, best.total_ped_trips),
            ("Transit", baseline.total_transit_trips, best.total_transit_trips),
        ]:
            base_share = 100.0 * base_val / baseline_total
            best_share = 100.0 * best_val / best_total
            change = best_share - base_share
            
            print(f"{mode:<25} {base_share:>14.2f}% {best_share:>14.2f}% {change:>+14.2f}pp")
        
        print("=" * 80)


def print_policy_comparison(
    baseline_policy: Policy,
    best_policy: Policy,
) -> None:
    """
    Print policy parameter comparison.
    
    Args:
        baseline_policy: Baseline policy
        best_policy: Optimized policy
    """
    print("\nPOLICY PARAMETERS")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 80)
    
    for key in ["transit_frequency_mult", "congestion_toll", "road_capacity_scale"]:
        base_val = getattr(baseline_policy, key)
        best_val = getattr(best_policy, key)
        change = best_val - base_val
        
        print(f"{key:<30} {base_val:>15.3f} {best_val:>15.3f} {change:>+15.3f}")
    
    print("-" * 80 + "\n")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare baseline vs optimized policy"
    )
    
    parser.add_argument(
        "--best-json",
        type=str,
        default="data/opt/best_policy.json",
        help="Path to best policy JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for evaluation (default: 123)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=30,
        help="Number of simulation ticks (default: 30)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-tick CSV files"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save comparison JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/opt/eval",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for policy evaluation."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Define baseline policy
    baseline_policy = Policy(
        transit_frequency_mult=1.0,
        congestion_toll=0.0,
        road_capacity_scale=1.0,
    )
    
    # Load best policy
    try:
        best_policy = load_best_policy(args.best_json)
        logger.info(f"Loaded best policy from {args.best_json}")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Using baseline policy as 'best' policy")
        best_policy = baseline_policy
    
    # Print policies
    print_policy_comparison(baseline_policy, best_policy)
    
    # Run evaluations
    logger.info(f"Running baseline evaluation ({args.ticks} ticks)...")
    baseline_summary, baseline_ticks = run_episode(
        baseline_policy,
        seed=args.seed,
        ticks=args.ticks,
    )
    
    logger.info(f"Running optimized policy evaluation ({args.ticks} ticks)...")
    best_summary, best_ticks = run_episode(
        best_policy,
        seed=args.seed,
        ticks=args.ticks,
    )
    
    # Print comparison
    print_comparison(baseline_summary, best_summary)
    
    # Save results
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        save_tick_csv(
            os.path.join(args.output_dir, "baseline_ticks.csv"),
            baseline_ticks
        )
        save_tick_csv(
            os.path.join(args.output_dir, "best_ticks.csv"),
            best_ticks
        )
    
    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        save_comparison_json(
            os.path.join(args.output_dir, "comparison.json"),
            baseline_policy,
            best_policy,
            baseline_summary,
            best_summary,
        )


if __name__ == "__main__":
    main()