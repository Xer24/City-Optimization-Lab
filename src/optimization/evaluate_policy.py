from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# --- Make "models", "simulation", "data" importable when running with PYTHONPATH=src ---
# (You run: PYTHONPATH=src python src/optimization/evaluate_policy.py ...)
from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel


# -----------------------------
# Policy + Metrics
# -----------------------------
@dataclass
class Policy:
    transit_frequency_mult: float = 1.0
    congestion_toll: float = 0.0
    road_capacity_scale: float = 1.0


@dataclass
class TickMetrics:
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
    avg_travel_time: float
    avg_energy: float
    mean_congestion: float
    congestion_p95: float


# -----------------------------
# Helpers copied from your main() style
# -----------------------------
def mean_flow(edge_flows: Dict[Any, float]) -> float:
    if not edge_flows:
        return 0.0
    vals = np.fromiter(edge_flows.values(), dtype=float)
    return float(vals.mean()) if vals.size else 0.0


def p95_flow(edge_flows: Dict[Any, float]) -> float:
    if not edge_flows:
        return 0.0
    vals = np.fromiter(edge_flows.values(), dtype=float)
    return float(np.percentile(vals, 95)) if vals.size else 0.0


def apply_capacity_scale(city: CityGrid, scale: float) -> None:
    """
    Optional: only affects sim if edges have 'capacity' attribute.
    Safe even if they don't.
    """
    for u, v, data in city.graph.edges(data=True):
        if "capacity" in data and data["capacity"] is not None:
            data["capacity"] = float(data["capacity"]) * float(scale)


def apply_policy_to_mode_shares(traffic: TrafficModel, policy: Policy) -> None:
    """
    Same mapping used in optimizer: policy -> (car/transit/ped) shares.
    """
    car = float(getattr(traffic, "car_share", 0.5))
    ped = float(getattr(traffic, "ped_share", 0.25))
    transit = float(getattr(traffic, "transit_share", 0.25))

    car_shift = 0.05 * (policy.congestion_toll / 5.0)              # [0, 0.05]
    transit_shift = 0.08 * (policy.transit_frequency_mult - 1.0)   # ~[-0.024, +0.04]

    car = car * (1.0 - car_shift)
    transit = transit * (1.0 + max(-0.3, min(0.5, transit_shift)))

    car = max(0.05, min(0.9, car))
    transit = max(0.05, min(0.9, transit))

    ped = max(0.05, 1.0 - car - transit)

    total = car + transit + ped
    traffic.car_share = float(car / total)
    traffic.transit_share = float(transit / total)
    traffic.ped_share = float(ped / total)


# -----------------------------
# Core evaluation run
# -----------------------------
def run_episode(policy: Policy, seed: int, ticks: int) -> Tuple[SummaryMetrics, List[TickMetrics]]:
    cfg = Sim_Config

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

    apply_capacity_scale(city, policy.road_capacity_scale)

    sim = SimulationEngine(city)

    traffic = TrafficModel(
        grid=city,
        trips_per_person=cfg.traffic.trips_per_person,
        rng_seed=seed,
    )
    apply_policy_to_mode_shares(traffic, policy)

    tick_rows: List[TickMetrics] = []

    for _ in range(ticks):
        current_tick = int(sim.tick)

        flows = traffic.run_multimodal_assignment(deterministic_demand=True)

        car_mean = mean_flow(flows.get("car", {}))
        ped_mean = mean_flow(flows.get("ped", {}))
        transit_mean = mean_flow(flows.get("transit", {}))

        mean_cong = (car_mean + ped_mean + transit_mean) / 3.0
        p95_cong = max(
            p95_flow(flows.get("car", {})),
            p95_flow(flows.get("ped", {})),
            p95_flow(flows.get("transit", {})),
        )

        # same proxy used in optimizer (replace later with real travel time)
        travel_time_proxy = 1.2 * car_mean + 0.8 * transit_mean + 0.6 * ped_mean

        grid = sim.energy.daily_grid()
        energy = float(grid.sum())

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

        sim.step()

    # aggregate summary
    avg_travel_time = float(np.mean([r.travel_time_proxy for r in tick_rows])) if tick_rows else 0.0
    avg_energy = float(np.mean([r.energy for r in tick_rows])) if tick_rows else 0.0
    mean_congestion = float(np.mean([r.mean_cong for r in tick_rows])) if tick_rows else 0.0
    congestion_p95 = float(np.max([r.p95_cong for r in tick_rows])) if tick_rows else 0.0

    return (
        SummaryMetrics(
            avg_travel_time=avg_travel_time,
            avg_energy=avg_energy,
            mean_congestion=mean_congestion,
            congestion_p95=congestion_p95,
        ),
        tick_rows,
    )


# -----------------------------
# I/O
# -----------------------------
def load_best_policy(best_json_path: str) -> Policy:
    with open(best_json_path, "r") as f:
        data = json.load(f)

    bp = data.get("best_policy", data)  # allow direct policy json too

    return Policy(
        transit_frequency_mult=float(bp.get("transit_frequency_mult", 1.0)),
        congestion_toll=float(bp.get("congestion_toll", 0.0)),
        road_capacity_scale=float(bp.get("road_capacity_scale", 1.0)),
    )


def save_tick_csv(path: str, rows: List[TickMetrics]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(path, index=False)


def print_comparison(baseline: SummaryMetrics, best: SummaryMetrics) -> None:
    def fmt(x: float) -> str:
        return f"{x:.4f}"

    print("\n=== Policy Evaluation Summary ===")
    print(f"{'Metric':<20} {'Baseline':>12} {'Best':>12} {'Delta (Best-BL)':>16}")
    print("-" * 64)

    for name in ["avg_travel_time", "avg_energy", "mean_congestion", "congestion_p95"]:
        bl = getattr(baseline, name)
        be = getattr(best, name)
        print(f"{name:<20} {fmt(bl):>12} {fmt(be):>12} {fmt(be - bl):>16}")

    print("")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs best policy over longer horizon.")
    p.add_argument("--best-json", type=str, default="data/opt/best_policy.json")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--save-csv", action="store_true", help="Save per-tick CSVs in data/opt/eval/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_policy = Policy(
        transit_frequency_mult=1.0,
        congestion_toll=0.0,
        road_capacity_scale=1.0,
    )

    best_policy = load_best_policy(args.best_json)

    print("Baseline policy:", baseline_policy)
    print("Best policy:    ", best_policy)

    baseline_summary, baseline_ticks = run_episode(baseline_policy, seed=args.seed, ticks=args.ticks)
    best_summary, best_ticks = run_episode(best_policy, seed=args.seed, ticks=args.ticks)

    print_comparison(baseline_summary, best_summary)

    if args.save_csv:
        save_tick_csv("data/opt/eval/baseline_ticks.csv", baseline_ticks)
        save_tick_csv("data/opt/eval/best_ticks.csv", best_ticks)
        print("Saved per-tick CSVs -> data/opt/eval/")


if __name__ == "__main__":
    main()
