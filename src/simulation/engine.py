"""
Core simulation engine.

Coordinates temporal evolution of urban systems including energy,
traffic, and other dynamic models. Provides centralized state
management and tick-based progression.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.models.city_grid import CityGrid
from src.models.energy_model import EnergyModel
from src.config import Sim_Config

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """
    Container for simulation state at a given tick.
    
    Attributes:
        tick: Current simulation tick (1 tick = 24 hours)
        total_energy: Total energy consumption this tick
        peak_energy_hour: Hour with peak energy demand
        metrics: Additional custom metrics
    """
    tick: int
    total_energy: float
    peak_energy_hour: int
    metrics: Dict[str, Any]


class SimulationEngine:
    """
    Main simulation engine for urban system dynamics.
    
    Coordinates multiple submodels (energy, traffic, etc.) over discrete
    time steps. Each tick represents one simulated day (24 hours).
    
    Attributes:
        city: CityGrid with network topology
        energy: EnergyModel for electricity demand
        tick: Current simulation tick (0-indexed)
        history: List of SimulationState objects for each tick
    
    Example:
        >>> from models.city_grid import CityGrid
        >>> city = CityGrid(width=10, height=10, seed=42)
        >>> sim = SimulationEngine(city)
        >>> 
        >>> # Run simulation
        >>> for _ in range(10):
        ...     state = sim.step()
        ...     print(f"Tick {state.tick}: Energy = {state.total_energy:.2f}")
        >>> 
        >>> # Analyze history
        >>> avg_energy = sim.get_average_metric("total_energy")
        >>> print(f"Average energy: {avg_energy:.2f}")
    """
    
    def __init__(
        self,
        city: CityGrid,
        config: Optional[Any] = None,
    ):
        """
        Initialize simulation engine.
        
        Args:
            city: CityGrid instance with network topology
            config: Configuration object (defaults to Sim_Config)
        """
        if config is None:
            config = Sim_Config
        
        self.city = city
        self.config = config
        self.tick = 0
        self.history: List[SimulationState] = []
        
        # Initialize energy model
        self.energy = EnergyModel(
            city=city,
            noise_std=config.energy.noise_std,
            rng_seed=config.seed + config.energy.rng_offset,
        )
        
        logger.info(
            f"Initialized SimulationEngine: "
            f"{city.graph.number_of_nodes()} nodes, "
            f"{city.graph.number_of_edges()} edges"
        )
    
    def step(self) -> SimulationState:
        """
        Execute one simulation tick (24-hour period).
        
        Updates all submodels and advances simulation clock.
        
        Returns:
            SimulationState with metrics from this tick
        
        Example:
            >>> state = sim.step()
            >>> print(f"Tick {state.tick}: {state.total_energy:.2f} kWh")
        """
        # Advance clock FIRST
        self.tick += 1
        
        # Simulate energy demand for all 24 hours
        demand_by_hour = self.energy.simulate_day(range(24))  # Returns numpy array
        
        # Compute aggregate metrics
        total_energy = float(demand_by_hour.sum())
        peak_hour = int(demand_by_hour.argmax())
        
        # Store state with CURRENT tick
        state = SimulationState(
            tick=self.tick,
            total_energy=total_energy,
            peak_energy_hour=peak_hour,
            metrics={
                "demand_by_hour": demand_by_hour.tolist(),  # Convert to list for serialization
            },
        )
        
        self.history.append(state)
        
        logger.debug(
            f"Tick {state.tick}: energy={total_energy:.2f}, "
            f"peak_hour={peak_hour}"
        )
        
        return state
    
    def run(self, n_ticks: int) -> List[SimulationState]:
        """
        Run simulation for multiple ticks.
        
        Args:
            n_ticks: Number of ticks to simulate
        
        Returns:
            List of SimulationState objects
        
        Example:
            >>> states = sim.run(n_ticks=30)  # 30 days
            >>> energies = [s.total_energy for s in states]
        """
        logger.info(f"Running simulation for {n_ticks} ticks")
        
        states = []
        for i in range(n_ticks):
            state = self.step()
            states.append(state)
            
            if (i + 1) % max(1, n_ticks // 10) == 0:
                logger.info(f"Progress: {i+1}/{n_ticks} ticks complete")
        
        logger.info(f"Simulation complete: {n_ticks} ticks")
        return states
    
    def reset(self) -> None:
        """
        Reset simulation to initial state.
        
        Clears history and resets tick counter to 0.
        Reinitializes energy model with same parameters.
        """
        self.tick = 0
        self.history.clear()
        
        # Reinitialize energy model
        self.energy = EnergyModel(
            city=self.city,
            noise_std=self.config.energy.noise_std,
            rng_seed=self.config.seed + self.config.energy.rng_offset,
        )
        
        logger.info("Simulation reset to initial state")
    
    def get_current_state(self) -> Optional[SimulationState]:
        """
        Get most recent simulation state.
        
        Returns:
            Latest SimulationState, or None if no ticks have run
        """
        return self.history[-1] if self.history else None
    
    def get_state_at_tick(self, tick: int) -> Optional[SimulationState]:
        """
        Retrieve state from specific tick.
        
        Args:
            tick: Tick number to retrieve
        
        Returns:
            SimulationState at that tick, or None if not found
        """
        for state in self.history:
            if state.tick == tick:
                return state
        return None
    
    def get_average_metric(self, metric_name: str) -> float:
        """
        Compute average value of a metric over all ticks.
        
        Args:
            metric_name: Name of metric (e.g., "total_energy")
        
        Returns:
            Average value across all recorded ticks
        
        Example:
            >>> avg_energy = sim.get_average_metric("total_energy")
            >>> avg_peak = sim.get_average_metric("peak_energy_hour")
        """
        if not self.history:
            return 0.0
        
        values = [getattr(state, metric_name) for state in self.history]
        return float(sum(values)) / len(values)
    
    def get_metric_series(self, metric_name: str) -> List[float]:
        """
        Extract time series of a metric.
        
        Args:
            metric_name: Name of metric to extract
        
        Returns:
            List of metric values, one per tick
        
        Example:
            >>> energy_series = sim.get_metric_series("total_energy")
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(energy_series)
            >>> plt.xlabel("Tick")
            >>> plt.ylabel("Energy (kWh)")
        """
        return [getattr(state, metric_name) for state in self.history]
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, min, max for each metric
        
        Example:
            >>> stats = sim.get_summary_statistics()
            >>> print(f"Mean energy: {stats['total_energy_mean']:.2f}")
            >>> print(f"Peak energy: {stats['total_energy_max']:.2f}")
        """
        if not self.history:
            return {}
        
        import numpy as np
        
        stats = {}
        
        # Total energy statistics
        energy_series = self.get_metric_series("total_energy")
        stats["total_energy_mean"] = float(np.mean(energy_series))
        stats["total_energy_std"] = float(np.std(energy_series))
        stats["total_energy_min"] = float(np.min(energy_series))
        stats["total_energy_max"] = float(np.max(energy_series))
        
        # Peak hour statistics
        peak_hours = self.get_metric_series("peak_energy_hour")
        stats["peak_hour_mode"] = float(max(set(peak_hours), key=peak_hours.count))
        
        # Simulation metadata
        stats["n_ticks"] = len(self.history)
        
        return stats
    
    def export_history_to_dict(self) -> List[Dict[str, Any]]:
        """
        Export simulation history as list of dictionaries.
        
        Useful for saving to JSON or converting to DataFrame.
        
        Returns:
            List of dictionaries, one per tick
        
        Example:
            >>> import pandas as pd
            >>> history = sim.export_history_to_dict()
            >>> df = pd.DataFrame(history)
            >>> df.to_csv("simulation_history.csv", index=False)
        """
        result = []
        
        for state in self.history:
            row = {
                "tick": state.tick,
                "total_energy": state.total_energy,
                "peak_energy_hour": state.peak_energy_hour,
            }
            # Add any additional metrics
            row.update(state.metrics)
            result.append(row)
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SimulationEngine("
            f"tick={self.tick}, "
            f"nodes={self.city.graph.number_of_nodes()}, "
            f"edges={self.city.graph.number_of_edges()})"
        )


def create_default_simulation(
    width: int = 20,
    height: int = 20,
    seed: int = 42,
) -> SimulationEngine:
    """
    Create simulation with default configuration.
    
    Convenience function for quick setup.
    
    Args:
        width: Grid width
        height: Grid height
        seed: Random seed
    
    Returns:
        Initialized SimulationEngine
    
    Example:
        >>> sim = create_default_simulation(width=15, height=15, seed=123)
        >>> states = sim.run(n_ticks=20)
    """
    city = CityGrid(
        width=width,
        height=height,
        seed=seed,
    )
    
    return SimulationEngine(city)