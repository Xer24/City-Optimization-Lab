"""
Policy optimization for urban transportation systems.

Implements optimization algorithms to find best policy parameters for
minimizing congestion, energy usage, and travel times. Uses simulation-based
evaluation with configurable objectives and search strategies.

Algorithms:
- Random search with local refinement
- Bayesian optimization (optional)
- Genetic algorithm (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

import numpy as np

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Sim_Config
from models.city_grid import CityGrid
from simulation.engine import SimulationEngine
from models.traffic_model import TrafficModel

logger = logging.getLogger(__name__)


# ============================================================
# Policy Definition
# ============================================================

@dataclass
class Policy:
    """
    Urban transportation policy parameters.
    
    Attributes:
        transit_frequency_mult: Transit service frequency multiplier (0.7-1.5)
        congestion_toll: Road pricing in $/trip (0.0-5.0)
        road_capacity_scale: Road capacity scaling factor (0.8-1.2)
    
    Example:
        >>> policy = Policy(transit_frequency_mult=1.3, congestion_toll=2.5)
        >>> print(policy)
    """
    transit_frequency_mult: float = 1.0
    congestion_toll: float = 0.0
    road_capacity_scale: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate policy parameters are within bounds."""
        for key, (lo, hi) in POLICY_BOUNDS.items():
            value = getattr(self, key)
            if not (lo <= value <= hi):
                logger.warning(
                    f"Policy parameter {key}={value:.3f} outside bounds [{lo}, {hi}]"
                )


# Policy bounds for search space
POLICY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "transit_frequency_mult": (0.70, 1.50),
    "congestion_toll": (0.00, 5.00),
    "road_capacity_scale": (0.80, 1.20),
}


def sample_policy(rng: np.random.Generator) -> Policy:
    """
    Sample random policy from uniform distribution over bounds.
    
    Args:
        rng: NumPy random number generator
    
    Returns:
        Random policy within bounds
    """
    vals = {}
    for key, (lo, hi) in POLICY_BOUNDS.items():
        vals[key] = float(rng.uniform(lo, hi))
    return Policy(**vals)


def clip_policy(policy: Policy) -> Policy:
    """
    Clip policy parameters to stay within bounds.
    
    Args:
        policy: Input policy (may be outside bounds)
    
    Returns:
        New policy with clipped parameters
    """
    d = asdict(policy)
    for key, (lo, hi) in POLICY_BOUNDS.items():
        d[key] = float(max(lo, min(hi, d[key])))
    return Policy(**d)


def mutate_policy(
    policy: Policy,
    rng: np.random.Generator,
    sigma: float = 0.06,
) -> Policy:
    """
    Mutate policy by adding Gaussian noise.
    
    Used for local refinement and genetic algorithms.
    
    Args:
        policy: Base policy
        rng: Random number generator
        sigma: Mutation strength (fraction of parameter range)
    
    Returns:
        Mutated policy (clipped to bounds)
    
    Example:
        >>> base = Policy(transit_frequency_mult=1.2)
        >>> rng = np.random.default_rng(42)
        >>> mutated = mutate_policy(base, rng, sigma=0.1)
    """
    d = asdict(policy)
    for key, (lo, hi) in POLICY_BOUNDS.items():
        span = hi - lo
        noise = rng.normal(0.0, sigma * span)
        d[key] = float(d[key] + noise)
    return clip_policy(Policy(**d))


def crossover_policies(
    parent1: Policy,
    parent2: Policy,
    rng: np.random.Generator,
) -> Policy:
    """
    Create child policy via uniform crossover.
    
    Args:
        parent1: First parent policy
        parent2: Second parent policy
        rng: Random number generator
    
    Returns:
        Child policy with mixed parameters
    """
    d1 = asdict(parent1)
    d2 = asdict(parent2)
    child = {}
    
    for key in POLICY_BOUNDS.keys():
        # Randomly choose from parent1 or parent2
        child[key] = d1[key] if rng.random() < 0.5 else d2[key]
    
    return Policy(**child)


# ============================================================
# Metrics and Objectives
# ============================================================

@dataclass
class Metrics:
    """
    Simulation performance metrics.
    
    Attributes:
        avg_travel_time: Average travel time proxy
        total_energy: Total energy consumption
        mean_congestion: Mean edge congestion
        congestion_p95: 95th percentile congestion
        total_emissions: Total emissions (if available)
        failed: Whether simulation failed (1) or succeeded (0)
    """
    avg_travel_time: float
    total_energy: float
    mean_congestion: float
    congestion_p95: float
    total_emissions: float = 0.0
    failed: int = 0


@dataclass
class ObjectiveConfig:
    """
    Objective function configuration (weights for each metric).
    
    Lower scores are better. Weights determine relative importance.
    
    Attributes:
        w_time: Weight for travel time
        w_energy: Weight for energy consumption
        w_mean_cong: Weight for mean congestion
        w_p95_cong: Weight for peak congestion
        w_emissions: Weight for emissions
        fail_penalty: Score for failed simulations
    
    Example:
        >>> # Prioritize congestion reduction
        >>> config = ObjectiveConfig(w_p95_cong=2.0, w_time=0.5)
    """
    w_time: float = 1.0
    w_energy: float = 0.25
    w_mean_cong: float = 0.6
    w_p95_cong: float = 1.2
    w_emissions: float = 0.0
    fail_penalty: float = 1e6


def objective(metrics: Metrics, config: ObjectiveConfig) -> float:
    """
    Compute scalar objective score from metrics.
    
    Args:
        metrics: Simulation performance metrics
        config: Objective weights
    
    Returns:
        Weighted objective score (lower is better)
    
    Example:
        >>> m = Metrics(avg_travel_time=100, total_energy=50, 
        ...             mean_congestion=20, congestion_p95=80)
        >>> cfg = ObjectiveConfig()
        >>> score = objective(m, cfg)
    """
    if metrics.failed:
        return config.fail_penalty
    
    score = (
        config.w_time * metrics.avg_travel_time
        + config.w_energy * metrics.total_energy
        + config.w_mean_cong * metrics.mean_congestion
        + config.w_p95_cong * metrics.congestion_p95
        + config.w_emissions * metrics.total_emissions
    )
    
    return float(score)


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


def apply_capacity_scale(city: CityGrid, scale: float) -> None:
    """
    Scale road capacities by factor.
    
    Only affects edges that have 'capacity' attribute.
    
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
    
    Maps policy knobs to behavioral changes:
    - Higher congestion toll → fewer car trips
    - Higher transit frequency → more transit trips
    - Pedestrian share adjusts to maintain total = 1.0
    
    Args:
        traffic: TrafficModel instance
        policy: Policy to apply
    
    Note:
        This is a simplified behavioral model. Real-world elasticities
        would require calibration to observed data.
    """
    # Start from current shares
    car = float(traffic.car_share)
    ped = float(traffic.ped_share)
    transit = float(traffic.transit_share)
    
    # Policy effects (calibrated constants)
    # Toll: $5 toll reduces car share by ~5 percentage points
    car_shift = 0.05 * (policy.congestion_toll / 5.0)
    
    # Transit frequency: +50% service increases transit by ~4 percentage points
    transit_shift = 0.08 * (policy.transit_frequency_mult - 1.0)
    
    # Apply shifts
    car = car * (1.0 - car_shift)
    transit = transit * (1.0 + max(-0.3, min(0.5, transit_shift)))
    
    # Clip to reasonable bounds
    car = max(0.05, min(0.9, car))
    transit = max(0.05, min(0.9, transit))
    
    # Pedestrian takes remainder
    ped = max(0.05, 1.0 - car - transit)
    
    # Renormalize
    total = car + transit + ped
    traffic.car_share = float(car / total)
    traffic.transit_share = float(transit / total)
    traffic.ped_share = float(ped / total)
    
    logger.debug(
        f"Applied policy: car={traffic.car_share:.2%}, "
        f"transit={traffic.transit_share:.2%}, "
        f"ped={traffic.ped_share:.2%}"
    )


# ============================================================
# Core Evaluation Function
# ============================================================

def run_simulation(
    policy: Policy,
    seed: int,
    num_ticks: int = 6,
) -> Metrics:
    """
    Run headless simulation to evaluate policy.
    
    Args:
        policy: Policy parameters to evaluate
        seed: Random seed for reproducibility
        num_ticks: Number of simulation ticks
    
    Returns:
        Performance metrics
    
    Example:
        >>> policy = Policy(transit_frequency_mult=1.3)
        >>> metrics = run_simulation(policy, seed=42, num_ticks=10)
        >>> print(f"Travel time: {metrics.avg_travel_time:.2f}")
    """
    try:
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
        
        # Apply policy: road capacity
        apply_capacity_scale(city, policy.road_capacity_scale)
        
        # Initialize simulation
        sim = SimulationEngine(city)
        
        # Initialize traffic model
        traffic = TrafficModel(
            grid=city,
            trips_per_person=cfg.traffic.trips_per_person,
            rng_seed=seed,
        )
        
        # Apply policy: mode shares
        apply_policy_to_mode_shares(traffic, policy)
        
        # Run simulation and collect metrics
        energy_list: List[float] = []
        mean_cong_list: List[float] = []
        p95_cong_list: List[float] = []
        time_proxy_list: List[float] = []
        
        for tick in range(num_ticks):
            # Run traffic assignment
            flows = traffic.run_multimodal_assignment(deterministic_demand=True)
            
            # Compute congestion metrics per mode
            car_mean = mean_flow(flows.get("car", {}))
            ped_mean = mean_flow(flows.get("ped", {}))
            transit_mean = mean_flow(flows.get("transit", {}))
            
            # Aggregate congestion
            mean_cong = (car_mean + ped_mean + transit_mean) / 3.0
            p95_cong = max(
                p95_flow(flows.get("car", {})),
                p95_flow(flows.get("ped", {})),
                p95_flow(flows.get("transit", {})),
            )
            
            # Travel time proxy (weighted by mode importance)
            time_proxy = 1.2 * car_mean + 0.8 * transit_mean + 0.6 * ped_mean
            
            # Energy consumption
            grid = sim.energy.daily_grid()
            energy_usage = float(grid.sum())
            
            # Store metrics
            mean_cong_list.append(float(mean_cong))
            p95_cong_list.append(float(p95_cong))
            time_proxy_list.append(float(time_proxy))
            energy_list.append(float(energy_usage))
            
            # Advance simulation
            sim.step()
            
            logger.debug(
                f"Tick {tick}: travel_time={time_proxy:.2f}, "
                f"energy={energy_usage:.2f}, congestion={mean_cong:.2f}"
            )
        
        # Aggregate over ticks
        metrics = Metrics(
            avg_travel_time=float(np.mean(time_proxy_list)),
            total_energy=float(np.mean(energy_list)),
            mean_congestion=float(np.mean(mean_cong_list)),
            congestion_p95=float(np.max(p95_cong_list)),
            total_emissions=0.0,
            failed=0,
        )
        
        logger.debug(f"Simulation complete: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        # Return failed metrics to avoid killing optimization
        return Metrics(
            avg_travel_time=0.0,
            total_energy=0.0,
            mean_congestion=0.0,
            congestion_p95=0.0,
            total_emissions=0.0,
            failed=1,
        )


# ============================================================
# Logging
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
    """Create CSV log file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def append_log(path: str, row: Dict[str, Any]) -> None:
    """Append row to CSV log file."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


def save_best(
    path: str,
    best_score: float,
    best_policy: Policy,
    best_metrics: Metrics,
    opt_cfg: OptConfig,
    obj_cfg: ObjectiveConfig,
) -> None:
    """
    Save best policy to JSON file.
    
    Args:
        path: Output JSON file path
        best_score: Best objective score
        best_policy: Best policy found
        best_metrics: Metrics for best policy
        opt_cfg: Optimization configuration
        obj_cfg: Objective configuration
    """
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
    
    logger.info(f"Saved best policy to {path}")


# ============================================================
# Optimization Configuration
# ============================================================

@dataclass
class OptConfig:
    """
    Optimization algorithm configuration.
    
    Attributes:
        n_trials: Number of random search trials
        seed: Random seed
        local_refine_steps: Number of local search steps
        local_sigma: Mutation strength for local search
        eval_repeats: Number of evaluations per policy (for robustness)
        num_ticks: Simulation length per evaluation
    """
    n_trials: int = 250
    seed: int = 42
    local_refine_steps: int = 80
    local_sigma: float = 0.06
    eval_repeats: int = 1
    num_ticks: int = 6


def evaluate_policy(
    policy: Policy,
    base_seed: int,
    opt_cfg: OptConfig,
    obj_cfg: ObjectiveConfig,
) -> Tuple[float, Metrics]:
    """
    Evaluate policy with multiple random seeds and average results.
    
    Args:
        policy: Policy to evaluate
        base_seed: Base random seed
        opt_cfg: Optimization configuration
        obj_cfg: Objective configuration
    
    Returns:
        Tuple of (average_score, representative_metrics)
    """
    scores = []
    metrics_rep = None
    
    for repeat in range(opt_cfg.eval_repeats):
        seed = base_seed + repeat
        metrics = run_simulation(policy, seed=seed, num_ticks=opt_cfg.num_ticks)
        score = objective(metrics, obj_cfg)
        scores.append(score)
        metrics_rep = metrics
    
    avg_score = float(np.mean(scores))
    
    return avg_score, metrics_rep if metrics_rep is not None else Metrics(
        0, 0, 0, 0, failed=1
    )


# ============================================================
# Optimization Algorithms
# ============================================================

def optimize(
    opt_cfg: OptConfig,
    obj_cfg: ObjectiveConfig,
    log_path: str,
    best_path: str,
) -> Dict[str, Any]:
    """
    Run optimization: random search + local refinement.
    
    Algorithm:
    1. Random search: sample n_trials policies uniformly
    2. Local refinement: hill-climb from best policy found
    
    Args:
        opt_cfg: Optimization configuration
        obj_cfg: Objective configuration
        log_path: CSV log file path
        best_path: Best policy JSON path
    
    Returns:
        Dictionary with optimization results
    
    Example:
        >>> opt_cfg = OptConfig(n_trials=100, local_refine_steps=50)
        >>> obj_cfg = ObjectiveConfig(w_time=1.0, w_p95_cong=1.5)
        >>> results = optimize(opt_cfg, obj_cfg, "runs.csv", "best.json")
    """
    rng = np.random.default_rng(opt_cfg.seed)
    ensure_logfile(log_path)
    
    best_score = float("inf")
    best_policy: Optional[Policy] = None
    best_metrics: Optional[Metrics] = None
    run_id = 0
    
    logger.info(
        f"Starting optimization: {opt_cfg.n_trials} random trials + "
        f"{opt_cfg.local_refine_steps} local steps"
    )
    
    # Phase 1: Random search
    logger.info("Phase 1: Random search")
    for trial in range(opt_cfg.n_trials):
        run_id += 1
        
        # Sample random policy
        policy = sample_policy(rng)
        
        # Evaluate
        score, metrics = evaluate_policy(
            policy,
            base_seed=opt_cfg.seed + run_id * 1000,
            opt_cfg=opt_cfg,
            obj_cfg=obj_cfg,
        )
        
        # Log result
        append_log(log_path, {
            "run_id": run_id,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "base_seed": opt_cfg.seed,
            "score": score,
            "failed": metrics.failed,
            **asdict(policy),
            **asdict(metrics),
        })
        
        # Update best
        if score < best_score:
            best_score = score
            best_policy = policy
            best_metrics = metrics
            save_best(best_path, best_score, best_policy, best_metrics, opt_cfg, obj_cfg)
            logger.info(f"New best at trial {run_id}: score={best_score:.6g}")
        
        # Progress update
        if run_id % max(1, opt_cfg.n_trials // 10) == 0:
            logger.info(
                f"Random search progress: {run_id}/{opt_cfg.n_trials}, "
                f"best={best_score:.6g}"
            )
    
    # Phase 2: Local refinement (hill climbing)
    if best_policy is not None and opt_cfg.local_refine_steps > 0:
        logger.info("Phase 2: Local refinement")
        current_policy = best_policy
        
        for step in range(opt_cfg.local_refine_steps):
            run_id += 1
            
            # Mutate current best
            candidate = mutate_policy(current_policy, rng, sigma=opt_cfg.local_sigma)
            
            # Evaluate
            score, metrics = evaluate_policy(
                candidate,
                base_seed=opt_cfg.seed + run_id * 1000,
                opt_cfg=opt_cfg,
                obj_cfg=obj_cfg,
            )
            
            # Log result
            append_log(log_path, {
                "run_id": run_id,
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "base_seed": opt_cfg.seed,
                "score": score,
                "failed": metrics.failed,
                **asdict(candidate),
                **asdict(metrics),
            })
            
            # Accept if better (greedy hill climbing)
            if score < best_score:
                best_score = score
                best_policy = candidate
                best_metrics = metrics
                current_policy = candidate
                save_best(best_path, best_score, best_policy, best_metrics, opt_cfg, obj_cfg)
                logger.info(f"Improved at step {step+1}: score={best_score:.6g}")
            
            # Progress update
            if (step + 1) % max(1, opt_cfg.local_refine_steps // 5) == 0:
                logger.info(
                    f"Local refinement: {step+1}/{opt_cfg.local_refine_steps}, "
                    f"best={best_score:.6g}"
                )
    
    logger.info(
        f"Optimization complete: best score={best_score:.6g} "
        f"after {run_id} evaluations"
    )
    
    return {
        "best_score": best_score,
        "best_policy": asdict(best_policy) if best_policy else None,
        "best_metrics": asdict(best_metrics) if best_metrics else None,
        "total_evaluations": run_id,
        "log_path": log_path,
        "best_path": best_path,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize urban transportation policies via simulation"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=250,
        help="Number of random search trials (default: 250)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--local-steps",
        type=int,
        default=80,
        help="Number of local refinement steps (default: 80)"
    )
    parser.add_argument(
        "--local-sigma",
        type=float,
        default=0.06,
        help="Mutation strength for local search (default: 0.06)"
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="Evaluations per policy for robustness (default: 1)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=6,
        help="Simulation ticks per evaluation (default: 6)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="data/opt/policy_runs.csv",
        help="Output CSV log path"
    )
    parser.add_argument(
        "--best",
        type=str,
        default="data/opt/best_policy.json",
        help="Output best policy JSON path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for optimization."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create optimization configuration
    opt_cfg = OptConfig(
        n_trials=args.trials,
        seed=args.seed,
        local_refine_steps=args.local_steps,
        local_sigma=args.local_sigma,
        eval_repeats=args.eval_repeats,
        num_ticks=args.ticks,
    )
    
    obj_cfg = ObjectiveConfig()
    
    logger.info("Configuration:")
    logger.info(f"  Optimization: {opt_cfg}")
    logger.info(f"  Objective: {obj_cfg}")
    
    # Run optimization
    result = optimize(
        opt_cfg=opt_cfg,
        obj_cfg=obj_cfg,
        log_path=args.log,
        best_path=args.best,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()