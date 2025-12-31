from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# --- Your project imports (same pattern as main.py) ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel


# ============================================================
# 1) Policy definition
# ============================================================

@dataclass
class Policy:
    # Keep it small and effective given what exists in your code right now
    transit_frequency_mult: float      # 0.70–1.50 (we map to mode shares)
    congestion_toll: float             # 0.00–5.00 (we map to mode shares)
    road_capacity_scale: float         # 0.80–1.20 (optional, only if edge has capacity attr)


POLICY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "transit_frequency_mult": (0.70, 1.50),
    "congestion_toll": (0.00, 5.00),
    "road_capacity_scale": (0.80, 1.20),
}


def sample_policy(rng: np.random.Generator) -> Policy:
    vals = {}
    for k, (lo, hi) in POLICY_BOUNDS.items():
        vals[k] = float(rng.uniform(lo, hi))
    return Policy(**vals)


def clip_policy(policy: Policy) -> Policy:
    d = asdict(policy)
    for k, (lo, hi) in POLICY_BOUNDS.items():
        d[k] = float(max(lo, min(hi, d[k])))
    return Policy(**d)


def mutate_policy(policy: Policy, rng: np.random.Generator, sigma: float = 0.06) -> Policy:
    d = asdict(policy)
    for k, (lo, hi) in POLICY_BOUNDS.items():
        span = hi - lo
        d[k] = float(d[k] + rng.normal(0.0, sigma * span))
    return clip_policy(Policy(**d))


# ============================================================
# 2) Metrics + default objective
# ============================================================

@dataclass
class Metrics:
    avg_travel_time: float
    total_energy: float
    mean_congestion: float
    congestion_p95: float
    total_emissions: float = 0.0
    failed: int = 0


@dataclass
class ObjectiveConfig:
    w_time: float = 1.0
    w_energy: float = 0.25
    w_mean_cong: float = 0.6
    w_p95_cong: float = 1.2
    w_emissions: float = 0.0
    fail_penalty: float = 1e6


def objective(m: Metrics, cfg: ObjectiveConfig) -> float:
    if m.failed:
        return cfg.fail_penalty
    return float(
        cfg.w_time * m.avg_travel_time
        + cfg.w_energy * m.total_energy
        + cfg.w_mean_cong * m.mean_congestion
        + cfg.w_p95_cong * m.congestion_p95
        + cfg.w_emissions * m.total_emissions
    )


# ============================================================
# 3) Logging (CSV)
# ============================================================

LOG_COLUMNS = [
    "run_id",
    "utc_time",
    "base_seed",
    "score",
    "failed",
    *POLICY_BOUNDS.keys(),
    "avg_travel_time",
    "total_energy",
    "mean_congestion",
    "congestion_p95",
    "total_emissions",
]


def ensure_logfile(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()


def append_log(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)


# ============================================================
# 4) Helper functions (metrics from your flows)
# ============================================================

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
    Optional: only does something if your edges have a 'capacity' attribute.
    Safe to call even if they don't.
    """
    G = city.graph
    for u, v, data in G.edges(data=True):
        if "capacity" in data and data["capacity"] is not None:
            data["capacity"] = float(data["capacity"]) * float(scale)


def apply_policy_to_mode_shares(
    traffic: TrafficModel,
    policy: Policy,
) -> None:
    """
    Map policy knobs -> mode shares (because this definitely exists today).

    Intuition:
    - Higher congestion_toll discourages car
    - Higher transit_frequency_mult encourages transit
    - Ped picks up leftover share automatically

    We keep shares normalized and clipped.
    """
    # Start from whatever TrafficModel initialized
    car = float(getattr(traffic, "car_share", 0.5))
    ped = float(getattr(traffic, "ped_share", 0.25))
    transit = float(getattr(traffic, "transit_share", 0.25))

    # Effects (tunable constants)
    # Toll effect: every +1 toll reduces car by ~3–6 percentage points
    car_shift = 0.05 * (policy.congestion_toll / 5.0)  # in [0, 0.05]
    # Transit frequency effect: boosts transit when >1, reduces when <1
    transit_shift = 0.08 * (policy.transit_frequency_mult - 1.0)  # about [-0.024, +0.04]

    car = car * (1.0 - car_shift)
    transit = transit * (1.0 + max(-0.3, min(0.5, transit_shift)))

    # Re-normalize: ped absorbs remainder (or give remainder proportionally)
    car = max(0.05, min(0.9, car))
    transit = max(0.05, min(0.9, transit))

    # Make ped the residual
    ped = max(0.05, 1.0 - car - transit)

    # If residual pushes total > 1 due to clipping, renormalize all
    total = car + transit + ped
    car /= total
    transit /= total
    ped /= total

    traffic.car_share = float(car)
    traffic.transit_share = float(transit)
    traffic.ped_share = float(ped)


# ============================================================
# 5) The ONLY required hook: run_simulation(policy, seed)
# ============================================================

def run_simulation(policy: Policy, seed: int, num_ticks: int = 6) -> Metrics:
    """
    Wraps your main() logic into a headless evaluation run.
    No plotting, no DB writes.

    Returns Metrics expected by optimizer.
    """
    try:
        cfg = Sim_Config

        # ---- Create city ----
        city = CityGrid(
            width=cfg.grid.width,
            height=cfg.grid.height,
            spacing=cfg.grid.spacing,
            diagonal=cfg.grid.diagonal,
            seed=seed,  # IMPORTANT: use optimizer seed here
            edge_keep=cfg.grid.edge_keep,
            diag_keep=cfg.grid.diag_keep,
            population_range=cfg.grid.population_range,
            density_range=cfg.grid.density_range,
            clusters_per_zone=cfg.grid.clusters_per_zone,
        )

        # ---- Apply policy (capacity scaling if available) ----
        apply_capacity_scale(city, policy.road_capacity_scale)

        # ---- Simulation engine ----
        sim = SimulationEngine(city)

        # ---- Traffic model ----
        traffic = TrafficModel(
            grid=city,
            trips_per_person=cfg.traffic.trips_per_person,
            rng_seed=seed,
        )

        # ---- Apply policy (mode shares definitely exist) ----
        apply_policy_to_mode_shares(traffic, policy)

        # ---- Run ticks, gather metrics ----
        energy_list: List[float] = []
        mean_cong_list: List[float] = []
        p95_cong_list: List[float] = []
        time_proxy_list: List[float] = []

        # Tick 0 + subsequent ticks
        for _ in range(num_ticks):
            # flows: dict with keys "car", "ped", "transit"
            flows = traffic.run_multimodal_assignment(deterministic_demand=True)

            # congestion proxies per mode
            car_mean = mean_flow(flows.get("car", {}))
            ped_mean = mean_flow(flows.get("ped", {}))
            transit_mean = mean_flow(flows.get("transit", {}))

            # combine into a single scalar congestion signal
            mean_cong = (car_mean + ped_mean + transit_mean) / 3.0
            p95_cong = max(
                p95_flow(flows.get("car", {})),
                p95_flow(flows.get("ped", {})),
                p95_flow(flows.get("transit", {})),
            )

            # travel time proxy (until you compute actual travel time)
            # Car congestion is often the big driver, so weight it higher
            time_proxy = 1.2 * car_mean + 0.8 * transit_mean + 0.6 * ped_mean

            # energy usage
            grid = sim.energy.daily_grid()
            energy_usage = float(grid.sum())

            mean_cong_list.append(float(mean_cong))
            p95_cong_list.append(float(p95_cong))
            time_proxy_list.append(float(time_proxy))
            energy_list.append(float(energy_usage))

            # advance sim clock
            sim.step()

        # Aggregate over ticks
        return Metrics(
            avg_travel_time=float(np.mean(time_proxy_list)),
            total_energy=float(np.mean(energy_list)),
            mean_congestion=float(np.mean(mean_cong_list)),
            congestion_p95=float(np.max(p95_cong_list)),  # worst peak over horizon
            total_emissions=0.0,
            failed=0,
        )

    except Exception:
        # If your sim ever errors on some policy, don’t kill optimization
        return Metrics(
            avg_travel_time=0.0,
            total_energy=0.0,
            mean_congestion=0.0,
            congestion_p95=0.0,
            total_emissions=0.0,
            failed=1,
        )


# ============================================================
# 6) Optimization loop
# ============================================================

@dataclass
class OptConfig:
    n_trials: int = 250
    seed: int = 42
    local_refine_steps: int = 80
    local_sigma: float = 0.06
    eval_repeats: int = 1
    num_ticks: int = 6


def evaluate_policy(pol: Policy, base_seed: int, opt_cfg: OptConfig, obj_cfg: ObjectiveConfig) -> Tuple[float, Metrics]:
    scores = []
    m_rep = None
    for r in range(opt_cfg.eval_repeats):
        m = run_simulation(pol, seed=base_seed + r, num_ticks=opt_cfg.num_ticks)
        s = objective(m, obj_cfg)
        scores.append(s)
        m_rep = m
    return float(np.mean(scores)), m_rep if m_rep is not None else Metrics(0, 0, 0, 0, failed=1)


def save_best(path: str, best_score: float, best_policy: Policy, best_metrics: Metrics, opt_cfg: OptConfig, obj_cfg: ObjectiveConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "best_score": best_score,
        "best_policy": asdict(best_policy),
        "best_metrics": asdict(best_metrics),
        "policy_bounds": POLICY_BOUNDS,
        "opt_config": asdict(opt_cfg),
        "objective_config": asdict(obj_cfg),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def optimize(opt_cfg: OptConfig, obj_cfg: ObjectiveConfig, log_path: str, best_path: str) -> Dict[str, Any]:
    rng = np.random.default_rng(opt_cfg.seed)
    ensure_logfile(log_path)

    best_score = float("inf")
    best_policy: Optional[Policy] = None
    best_metrics: Optional[Metrics] = None
    run_id = 0

    # Random search
    for _ in range(opt_cfg.n_trials):
        run_id += 1
        pol = sample_policy(rng)
        score, m = evaluate_policy(pol, base_seed=opt_cfg.seed + run_id * 1000, opt_cfg=opt_cfg, obj_cfg=obj_cfg)

        append_log(log_path, {
            "run_id": run_id,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "base_seed": opt_cfg.seed,
            "score": score,
            "failed": m.failed,
            **asdict(pol),
            **asdict(m),
        })

        if score < best_score:
            best_score, best_policy, best_metrics = score, pol, m
            save_best(best_path, best_score, best_policy, best_metrics, opt_cfg, obj_cfg)

        if run_id % max(1, opt_cfg.n_trials // 10) == 0:
            print(f"[random] {run_id}/{opt_cfg.n_trials} best={best_score:.6g}")

    # Local refine (hill climb)
    if best_policy is not None and opt_cfg.local_refine_steps > 0:
        current = best_policy
        print("\nStarting local refinement...")
        for i in range(opt_cfg.local_refine_steps):
            run_id += 1
            cand = mutate_policy(current, rng, sigma=opt_cfg.local_sigma)
            score, m = evaluate_policy(cand, base_seed=opt_cfg.seed + run_id * 1000, opt_cfg=opt_cfg, obj_cfg=obj_cfg)

            append_log(log_path, {
                "run_id": run_id,
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "base_seed": opt_cfg.seed,
                "score": score,
                "failed": m.failed,
                **asdict(cand),
                **asdict(m),
            })

            if score < best_score:
                best_score, best_policy, best_metrics = score, cand, m
                current = cand
                save_best(best_path, best_score, best_policy, best_metrics, opt_cfg, obj_cfg)

            if (i + 1) % max(1, opt_cfg.local_refine_steps // 5) == 0:
                print(f"[local] {i+1}/{opt_cfg.local_refine_steps} best={best_score:.6g}")

    return {
        "best_score": best_score,
        "best_policy": asdict(best_policy) if best_policy else None,
        "best_metrics": asdict(best_metrics) if best_metrics else None,
        "log_path": log_path,
        "best_path": best_path,
    }


# ============================================================
# 7) CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-steps", type=int, default=80)
    p.add_argument("--local-sigma", type=float, default=0.06)
    p.add_argument("--eval-repeats", type=int, default=1)
    p.add_argument("--ticks", type=int, default=6)
    p.add_argument("--log", type=str, default="data/opt/policy_runs.csv")
    p.add_argument("--best", type=str, default="data/opt/best_policy.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    opt_cfg = OptConfig(
        n_trials=args.trials,
        seed=args.seed,
        local_refine_steps=args.local_steps,
        local_sigma=args.local_sigma,
        eval_repeats=args.eval_repeats,
        num_ticks=args.ticks,
    )
    obj_cfg = ObjectiveConfig()

    result = optimize(opt_cfg, obj_cfg, log_path=args.log, best_path=args.best)
    print("\n=== Done ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
