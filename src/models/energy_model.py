"""
Energy demand modeling for urban simulation.

Implements hour-by-hour energy consumption based on:
- Zone-specific demand profiles (residential, commercial, industrial)
- Population and density effects
- Temporal patterns (morning/evening peaks, business hours)

Energy Formula:
    E(t) = baseline × zone(t) × (1 + α×pop/avg_pop) × (1 + β×density) × noise
    
Where:
    - baseline: Base power requirement for the node
    - zone(t): Time-varying zone multiplier (different for each zone type)
    - α: Population sensitivity parameter
    - β: Density sensitivity parameter
    - noise: Stochastic variation factor
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Iterable, List, Tuple, TYPE_CHECKING
import numpy as np
import random
import matplotlib.pyplot as plt
import logging

if TYPE_CHECKING:
    from models.city_grid import CityGrid

logger = logging.getLogger(__name__)

Node = Hashable


@dataclass
class DemandProfile:
    """
    24-hour energy demand profile for a zone type.
    
    Represents temporal variation in energy consumption throughout the day.
    Values are multipliers applied to baseline energy demand.
    
    Attributes:
        name: Profile identifier (e.g., "residential", "commercial")
        values: 24-element array of hourly demand multipliers
    
    Example:
        >>> # Peak demand at 6 PM (hour 18)
        >>> values = np.ones(24)
        >>> values[18] = 2.0
        >>> profile = DemandProfile("evening_peak", values)
        >>> profile.energy_factor(18)
        2.0
    """
    
    name: str
    values: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate profile data."""
        self.values = np.asarray(self.values, dtype=float)
        
        if len(self.values) != 24:
            raise ValueError(
                f"Demand profile must have 24 hourly values, got {len(self.values)}"
            )
        
        if np.any(self.values < 0):
            raise ValueError("Demand profile values must be non-negative")
    
    def energy_factor(self, hour: int) -> float:
        """
        Get demand multiplier for a specific hour.
        
        Args:
            hour: Hour of day (wraps using modulo 24)
        
        Returns:
            Energy demand multiplier for that hour
        """
        return float(self.values[int(hour) % 24])
    
    def peak_hour(self) -> int:
        """Return hour with maximum demand."""
        return int(np.argmax(self.values))
    
    def off_peak_hour(self) -> int:
        """Return hour with minimum demand."""
        return int(np.argmin(self.values))
    
    def average_factor(self) -> float:
        """Return average demand multiplier across all hours."""
        return float(np.mean(self.values))
    
    def peak_to_average_ratio(self) -> float:
        """
        Calculate peak-to-average demand ratio.
        
        Higher values indicate more pronounced demand peaks.
        """
        avg = self.average_factor()
        if avg == 0:
            return 0.0
        return float(np.max(self.values) / avg)


def default_profiles() -> Dict[str, DemandProfile]:
    """
    Construct default 24-hour demand profiles for each zone type.
    
    Profiles reflect typical usage patterns:
    - Residential: Morning (6-9 AM) and evening (3-10 PM) peaks
    - Commercial: Business hours (9 AM - 6 PM) peak
    - Industrial: Extended daytime operation (7 AM - 7 PM)
    
    Returns:
        Dictionary mapping zone names to DemandProfile objects
    
    Example:
        >>> profiles = default_profiles()
        >>> print(profiles["residential"].peak_hour())
        18  # 6 PM
    """
    hours = np.arange(24)
    
    # Residential: Morning and evening peaks
    residential = np.where(
        (6 <= hours) & (hours < 9),   # Morning: 6-9 AM
        1.3,
        np.where(
            (15 <= hours) & (hours < 22),  # Evening: 3-10 PM
            1.6,
            0.8  # Off-peak
        ),
    )
    
    # Commercial: Business hours peak
    commercial = np.where(
        (9 <= hours) & (hours < 18),   # 9 AM - 6 PM
        1.7,
        0.4  # Night/weekend-like
    )
    
    # Industrial: Extended daytime operation
    industrial = np.where(
        (7 <= hours) & (hours < 19),   # 7 AM - 7 PM
        1.2,
        0.9  # Reduced but still active
    )
    
    return {
        "residential": DemandProfile("residential", residential),
        "commercial": DemandProfile("commercial", commercial),
        "industrial": DemandProfile("industrial", industrial),
    }


def create_custom_profile(
    name: str,
    peak_hours: List[int],
    peak_factor: float = 1.5,
    base_factor: float = 0.8,
) -> DemandProfile:
    """
    Create a custom demand profile with specified peak hours.
    
    Args:
        name: Profile identifier
        peak_hours: List of hours (0-23) that should have peak demand
        peak_factor: Demand multiplier during peak hours
        base_factor: Demand multiplier during off-peak hours
    
    Returns:
        DemandProfile with specified peak pattern
    
    Example:
        >>> # Create profile with lunch hour peak
        >>> profile = create_custom_profile(
        ...     "restaurant",
        ...     peak_hours=[12, 13, 18, 19, 20],
        ...     peak_factor=2.0
        ... )
    """
    values = np.full(24, base_factor, dtype=float)
    for hour in peak_hours:
        if 0 <= hour < 24:
            values[hour] = peak_factor
    
    return DemandProfile(name, values)


class EnergyModel:
    """
    Compute hourly energy demand for city nodes based on zone, population, and density.
    
    Models energy consumption using a multiplicative formula that accounts for:
    - Baseline energy requirements (node attribute)
    - Zone-specific temporal patterns (residential vs. commercial vs. industrial)
    - Population effects (more people = more demand)
    - Density effects (denser areas = higher demand)
    - Stochastic noise (realistic variation)
    
    Attributes:
        city: CityGrid instance with node attributes
        graph: NetworkX graph from city (convenience reference)
        profiles: Demand profiles by zone type
        population_scale: Sensitivity to population (α parameter)
        density_scale: Sensitivity to density (β parameter)
        avg_pop: Average population across all nodes (for normalization)
        noise_std: Standard deviation of multiplicative noise
        rng: Random number generator for noise
    
    Example:
        >>> from models.city_grid import CityGrid
        >>> grid = CityGrid(width=20, height=20, seed=42)
        >>> energy = EnergyModel(grid, population_scale=0.5, density_scale=0.8)
        >>> daily_demand = energy.simulate_day()
        >>> print(f"Total daily energy: {daily_demand.sum():.1f}")
    """
    
    def __init__(
        self,
        city: "CityGrid",
        *,
        profiles: Optional[Dict[str, DemandProfile]] = None,
        population_scale: float = 0.5,
        density_scale: float = 0.8,
        noise_std: float = 0.1,
        rng_seed: Optional[int] = None,
    ):
        """
        Initialize energy model for a city grid.
        
        Args:
            city: CityGrid with population, density, and zoning attributes
            profiles: Custom demand profiles (uses defaults if None)
            population_scale: Population sensitivity (α), typically 0.3-0.7
            density_scale: Density sensitivity (β), typically 0.5-1.0
            noise_std: Std dev of multiplicative noise (0 = deterministic)
            rng_seed: Random seed for reproducible noise
        
        Raises:
            ValueError: If scale parameters are negative or noise_std invalid
        """
        # Validation
        if population_scale < 0:
            raise ValueError(f"population_scale must be non-negative, got {population_scale}")
        
        if density_scale < 0:
            raise ValueError(f"density_scale must be non-negative, got {density_scale}")
        
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        
        self.city = city
        self.graph = city.graph
        self.population_scale = float(population_scale)
        self.density_scale = float(density_scale)
        self.noise_std = float(noise_std)
        self.rng = random.Random(rng_seed)
        
        # Load or create profiles
        self.profiles = profiles if profiles is not None else default_profiles()
        
        # Compute average population for normalization
        populations = [
            float(data.get("population", 0.0))
            for _, data in self.graph.nodes(data=True)
        ]
        self.avg_pop = float(np.mean(populations)) if populations else 1.0
        
        logger.info(
            f"EnergyModel initialized: {len(self.graph.nodes)} nodes, "
            f"avg_pop={self.avg_pop:.1f}, "
            f"α={population_scale}, β={density_scale}"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnergyModel(nodes={self.graph.number_of_nodes()}, "
            f"α={self.population_scale:.2f}, β={self.density_scale:.2f})"
        )
    
    def node_demand(self, node: Node, hour: int) -> float:
        """
        Compute energy demand for a single node at a given hour.
        
        Applies the energy formula:
            E(t) = baseline × zone(t) × (1 + α×pop/avg_pop) × (1 + β×density) × noise
        
        Args:
            node: Node identifier
            hour: Hour of day (0-23, wraps via modulo)
        
        Returns:
            Energy demand in arbitrary units (consistent with baseline_energy)
        
        Example:
            >>> demand_morning = energy.node_demand((5, 5), hour=8)
            >>> demand_evening = energy.node_demand((5, 5), hour=20)
            >>> print(f"Evening/Morning ratio: {demand_evening/demand_morning:.2f}")
        """
        try:
            data = self.graph.nodes[node]
        except KeyError:
            logger.warning(f"Node {node} not found in graph")
            return 0.0
        
        # Extract node attributes
        baseline = float(data.get("baseline_energy", 0.0))
        zoning = str(data.get("zoning", "residential")).lower()
        population = float(data.get("population", 0.0))
        density = float(data.get("density", 0.0))
        
        # Zone-specific temporal factor
        profile = self.profiles.get(zoning)
        zone_factor = profile.energy_factor(hour) if profile else 1.0
        
        # Population factor (normalized by city average)
        if self.avg_pop > 0:
            pop_factor = 1.0 + self.population_scale * (population / self.avg_pop)
        else:
            pop_factor = 1.0
        
        # Density factor
        density_factor = 1.0 + self.density_scale * density
        
        # Compute base demand
        demand = baseline * zone_factor * pop_factor * density_factor
        
        # Add multiplicative noise
        if self.noise_std > 0:
            noise = self.rng.gauss(1.0, self.noise_std)
            demand *= max(noise, 0.1)  # Floor at 10% to avoid negative/zero
        
        return float(demand)
    
    def city_demand(self, hour: int) -> float:
        """
        Compute total city-wide energy demand at a given hour.
        
        Args:
            hour: Hour of day (0-23)
        
        Returns:
            Sum of demand across all nodes
        
        Example:
            >>> morning_demand = energy.city_demand(8)
            >>> evening_demand = energy.city_demand(20)
        """
        return float(sum(
            self.node_demand(node, hour)
            for node in self.graph.nodes
        ))
    
    def simulate_day(self, hours: Optional[Iterable[int]] = None) -> np.ndarray:
        """
        Simulate total city demand over multiple hours.
        
        Args:
            hours: Iterable of hours to simulate (default: range(24))
        
        Returns:
            Array of total city demand for each hour
        
        Example:
            >>> # Full 24-hour cycle
            >>> demand_24h = energy.simulate_day()
            >>> 
            >>> # Just business hours
            >>> demand_business = energy.simulate_day(range(9, 18))
            >>> 
            >>> # Multi-day simulation
            >>> demand_week = energy.simulate_day(range(24 * 7))
        """
        if hours is None:
            hours = range(24)
        
        hours_list = list(hours)
        demand = np.empty(len(hours_list), dtype=float)
        
        for i, h in enumerate(hours_list):
            demand[i] = self.city_demand(h)
        
        return demand
    
    def simulate_day_by_node(
        self,
        hours: Optional[Iterable[int]] = None,
    ) -> Dict[Node, np.ndarray]:
        """
        Simulate per-node energy demand over multiple hours.
        
        Useful for spatial analysis of energy consumption patterns.
        
        Args:
            hours: Hours to simulate (default: range(24))
        
        Returns:
            Dictionary mapping each node to array of hourly demands
        
        Example:
            >>> node_demands = energy.simulate_day_by_node()
            >>> 
            >>> # Find node with highest peak demand
            >>> peak_node = max(
            ...     node_demands.items(),
            ...     key=lambda x: x[1].max()
            ... )
            >>> print(f"Peak node: {peak_node[0]}, demand: {peak_node[1].max():.1f}")
        """
        if hours is None:
            hours = range(24)
        
        hours_list = list(hours)
        result: Dict[Node, np.ndarray] = {}
        
        for node in self.graph.nodes:
            demand_array = np.zeros(len(hours_list), dtype=float)
            for i, h in enumerate(hours_list):
                demand_array[i] = self.node_demand(node, h)
            result[node] = demand_array
        
        return result
    
    def simulate_by_zone(
        self,
        hours: Optional[Iterable[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate energy demand aggregated by zone type.
        
        Args:
            hours: Hours to simulate (default: range(24))
        
        Returns:
            Dictionary mapping zone type to hourly demand array
        
        Example:
            >>> by_zone = energy.simulate_by_zone()
            >>> print(f"Residential peak: {by_zone['residential'].max():.1f}")
            >>> print(f"Commercial peak: {by_zone['commercial'].max():.1f}")
        """
        if hours is None:
            hours = range(24)
        
        hours_list = list(hours)
        
        # Initialize result dict
        result = {zone: np.zeros(len(hours_list), dtype=float) for zone in self.profiles.keys()}
        
        # Aggregate by zone
        for node, data in self.graph.nodes(data=True):
            zone = data.get("zoning", "residential")
            if zone in result:
                for i, h in enumerate(hours_list):
                    result[zone][i] += self.node_demand(node, h)
        
        return result
    
    def daily_grid(self) -> np.ndarray:
        """
        Return 2D grid of total energy per node over 24 hours.
        
        Useful for spatial visualization of energy hotspots.
        
        Returns:
            height × width array where grid[row, col] = total daily energy
        
        Raises:
            ValueError: If city doesn't have height/width attributes
        
        Example:
            >>> grid = energy.daily_grid()
            >>> hotspot = np.unravel_index(grid.argmax(), grid.shape)
            >>> print(f"Energy hotspot at row {hotspot[0]}, col {hotspot[1]}")
        """
        height = getattr(self.city, "height", None)
        width = getattr(self.city, "width", None)
        
        if height is None or width is None:
            raise ValueError("CityGrid must have 'height' and 'width' attributes")
        
        grid = np.zeros((height, width), dtype=float)
        
        for node, data in self.graph.nodes(data=True):
            row = data.get("row")
            col = data.get("col")
            
            if row is None or col is None:
                continue
            
            # Sum demand over 24 hours
            daily_total = sum(self.node_demand(node, h) for h in range(24))
            grid[row, col] = daily_total
        
        return grid
    
    def get_peak_demand(self, hours: Optional[Iterable[int]] = None) -> Tuple[int, float]:
        """
        Find hour with maximum city-wide demand.
        
        Args:
            hours: Hours to check (default: range(24))
        
        Returns:
            Tuple of (peak_hour, peak_demand)
        
        Example:
            >>> hour, demand = energy.get_peak_demand()
            >>> print(f"Peak demand: {demand:.1f} at hour {hour}")
        """
        if hours is None:
            hours = range(24)
        
        demands = [(h, self.city_demand(h)) for h in hours]
        peak_hour, peak_demand = max(demands, key=lambda x: x[1])
        
        return int(peak_hour), float(peak_demand)
    
    def compute_load_factor(self, hours: Optional[Iterable[int]] = None) -> float:
        """
        Compute load factor (average demand / peak demand).
        
        Higher load factor indicates more efficient utilization.
        Typical values: 0.5-0.7 for cities with pronounced peaks.
        
        Args:
            hours: Hours to analyze (default: range(24))
        
        Returns:
            Load factor between 0 and 1
        """
        if hours is None:
            hours = range(24)
        
        demand = self.simulate_day(hours)
        
        if len(demand) == 0 or demand.max() == 0:
            return 0.0
        
        return float(demand.mean() / demand.max())
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_heatmap(
        self,
        *,
        ax=None,
        show: bool = True,
        cmap: str = "YlOrRd",
        title: Optional[str] = None,
    ) -> np.ndarray:
        """
        Plot spatial heatmap of total 24-hour energy demand.
        
        Args:
            ax: Matplotlib axes (creates new figure if None)
            show: Display plot immediately
            cmap: Colormap name (default: 'YlOrRd' - yellow to red)
            title: Custom title (default: auto-generated)
        
        Returns:
            The underlying height × width energy grid
        
        Example:
            >>> grid = energy.plot_heatmap(cmap='plasma')
            >>> print(f"Max daily demand: {grid.max():.1f}")
        """
        grid = self.daily_grid()
        
        # Create figure if needed
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True
        
        # Plot heatmap
        im = ax.imshow(grid, origin="lower", aspect="equal", cmap=cmap)
        
        # Labels
        if title is None:
            title = "Total 24-Hour Energy Demand"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Column", fontsize=10)
        ax.set_ylabel("Row", fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Energy (sum over 24 hrs)", fontsize=10)
        
        if created_fig and show:
            plt.tight_layout()
            plt.show()
        
        return grid
    
    def plot_temporal_profile(
        self,
        hours: Optional[Iterable[int]] = None,
        ax=None,
        show: bool = True,
        plot_by_zone: bool = False,
    ) -> None:
        """
        Plot temporal demand profile (total or by zone).
        
        Args:
            hours: Hours to plot (default: range(24))
            ax: Matplotlib axes
            show: Display immediately
            plot_by_zone: If True, plot separate curves for each zone type
        
        Example:
            >>> # Total city demand
            >>> energy.plot_temporal_profile()
            >>> 
            >>> # Demand by zone type
            >>> energy.plot_temporal_profile(plot_by_zone=True)
        """
        if hours is None:
            hours = range(24)
        
        hours_list = list(hours)
        
        # Create figure
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True
        
        if plot_by_zone:
            # Plot by zone
            by_zone = self.simulate_by_zone(hours_list)
            
            for zone, demand in by_zone.items():
                ax.plot(hours_list, demand, marker='o', label=zone.capitalize(), linewidth=2)
            
            ax.set_ylabel("Energy Demand by Zone", fontsize=11)
            ax.legend(loc='best', fontsize=10)
        else:
            # Plot total
            demand = self.simulate_day(hours_list)
            ax.plot(hours_list, demand, marker='o', color='darkblue', linewidth=2)
            ax.set_ylabel("Total Energy Demand", fontsize=11)
        
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_title("Energy Demand Profile", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if created_fig and show:
            plt.tight_layout()
            plt.show()
    
    def plot_combined_analysis(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        """
        Create comprehensive 2-panel energy analysis figure.
        
        Left panel: Spatial heatmap
        Right panel: Temporal profiles by zone
        
        Args:
            figsize: Figure dimensions (width, height)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Spatial heatmap
        self.plot_heatmap(ax=ax1, show=False)
        
        # Temporal profiles
        self.plot_temporal_profile(ax=ax2, show=False, plot_by_zone=True)
        
        plt.tight_layout()
        plt.show()