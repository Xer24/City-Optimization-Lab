"""
Multi-modal transportation flow data structures.

This module defines types and containers for managing traffic flows
across different transportation modes (car, pedestrian, public transit).
Provides utilities for aggregating, analyzing, and comparing flows.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Hashable, Literal, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Type aliases
Edge = Tuple[Hashable, Hashable]
TransportMode = Literal["car", "ped", "transit"]

# Constants for transport modes
TRANSPORT_MODES: Tuple[str, ...] = ("car", "ped", "transit")

# Legacy mode names for backward compatibility
LEGACY_MODE_NAMES = {
    "pedestrian": "ped",
    "public_transit": "transit",
}


@dataclass
class MultiModalFlows:
    """
    Container for traffic flows across multiple transportation modes.
    
    Stores edge-level flow data for each mode, enabling multi-modal
    traffic analysis, visualization, and policy evaluation.
    
    Attributes:
        car: Dictionary mapping edges to vehicle flow volumes
        ped: Dictionary mapping edges to pedestrian flow volumes  
        transit: Dictionary mapping edges to public transit flow volumes
    
    Example:
        >>> flows = MultiModalFlows(
        ...     car={(0, 1): 100.0, (1, 2): 50.0},
        ...     ped={(0, 1): 20.0},
        ...     transit={(1, 2): 30.0}
        ... )
        >>> total = flows.total()
        >>> total[(0, 1)]
        120.0
        >>> flows.get_mode_share("car")
        0.625
    
    Note:
        Mode names use abbreviated forms for consistency:
        - "car" for automobiles
        - "ped" for pedestrians
        - "transit" for public transportation
    """
    
    car: Dict[Edge, float]
    ped: Dict[Edge, float]
    transit: Dict[Edge, float]
    
    def __post_init__(self) -> None:
        """Validate flow dictionaries after initialization."""
        # Ensure all values are non-negative
        for mode_name, flow_dict in [
            ("car", self.car),
            ("ped", self.ped),
            ("transit", self.transit),
        ]:
            for edge, flow in flow_dict.items():
                if flow < 0:
                    raise ValueError(
                        f"{mode_name} flow on edge {edge} is negative: {flow}"
                    )
    
    def total(self) -> Dict[Edge, float]:
        """
        Aggregate flows across all modes.
        
        Returns:
            Dictionary mapping each edge to its total flow across all modes
        
        Example:
            >>> combined = flows.total()
            >>> max_edge = max(combined.items(), key=lambda x: x[1])
            >>> print(f"Busiest edge: {max_edge[0]} with {max_edge[1]:.0f} trips")
        """
        combined: Dict[Edge, float] = {}
        
        for flow_dict in (self.car, self.ped, self.transit):
            for edge, flow in flow_dict.items():
                combined[edge] = combined.get(edge, 0.0) + float(flow)
        
        return combined
    
    def get_mode_flows(self, mode: str) -> Dict[Edge, float]:
        """
        Get flows for a specific transportation mode.
        
        Args:
            mode: One of "car", "ped", "transit" (or legacy names)
        
        Returns:
            Dictionary of edge flows for the specified mode
        
        Raises:
            ValueError: If mode is not recognized
        
        Example:
            >>> car_flows = flows.get_mode_flows("car")
            >>> print(f"Car trips: {sum(car_flows.values()):.0f}")
        """
        # Handle legacy mode names
        mode = LEGACY_MODE_NAMES.get(mode, mode)
        
        mode_map = {
            "car": self.car,
            "ped": self.ped,
            "transit": self.transit,
        }
        
        if mode not in mode_map:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be one of: {list(mode_map.keys())}"
            )
        
        return mode_map[mode]
    
    def max_flow(self, mode: Optional[str] = None) -> float:
        """
        Find maximum flow value.
        
        Args:
            mode: Specific mode to check, or None for all modes
        
        Returns:
            Maximum flow value across specified mode(s)
        
        Example:
            >>> print(f"Max car flow: {flows.max_flow('car'):.0f}")
            >>> print(f"Max overall flow: {flows.max_flow():.0f}")
        """
        if mode is None:
            total_flows = self.total()
            return max(total_flows.values()) if total_flows else 0.0
        
        flows_dict = self.get_mode_flows(mode)
        return max(flows_dict.values()) if flows_dict else 0.0
    
    def total_volume(self, mode: Optional[str] = None) -> float:
        """
        Calculate total flow volume.
        
        Args:
            mode: Specific mode to sum, or None for all modes
        
        Returns:
            Sum of all flows for specified mode(s)
        
        Example:
            >>> total_trips = flows.total_volume()
            >>> car_trips = flows.total_volume("car")
            >>> print(f"Total: {total_trips:.0f}, Car: {car_trips:.0f}")
        """
        if mode is None:
            return sum(self.total().values())
        
        flows_dict = self.get_mode_flows(mode)
        return sum(flows_dict.values())
    
    def get_mode_share(self, mode: str) -> float:
        """
        Calculate modal share (fraction of total trips).
        
        Args:
            mode: Mode to calculate share for
        
        Returns:
            Fraction of trips using this mode (0 to 1)
        
        Example:
            >>> car_share = flows.get_mode_share("car")
            >>> print(f"Car mode share: {car_share:.1%}")
        """
        mode_volume = self.total_volume(mode)
        total_trips = self.total_volume()
        
        if total_trips == 0:
            return 0.0
        
        return mode_volume / total_trips
    
    def get_all_mode_shares(self) -> Dict[str, float]:
        """
        Calculate modal shares for all modes.
        
        Returns:
            Dictionary mapping each mode to its share (0 to 1)
        
        Example:
            >>> shares = flows.get_all_mode_shares()
            >>> for mode, share in shares.items():
            ...     print(f"{mode}: {share:.1%}")
        """
        return {
            "car": self.get_mode_share("car"),
            "ped": self.get_mode_share("ped"),
            "transit": self.get_mode_share("transit"),
        }
    
    def get_loaded_edges(self, mode: Optional[str] = None) -> int:
        """
        Count number of edges with positive flow.
        
        Args:
            mode: Specific mode, or None for any mode
        
        Returns:
            Number of edges carrying traffic
        """
        if mode is None:
            total_flows = self.total()
            return sum(1 for f in total_flows.values() if f > 0)
        
        flows_dict = self.get_mode_flows(mode)
        return sum(1 for f in flows_dict.values() if f > 0)
    
    def get_average_flow(self, mode: Optional[str] = None) -> float:
        """
        Calculate average flow on loaded edges.
        
        Args:
            mode: Specific mode, or None for all modes
        
        Returns:
            Average flow (excluding zero-flow edges)
        """
        if mode is None:
            total_flows = self.total()
            positive = [f for f in total_flows.values() if f > 0]
        else:
            flows_dict = self.get_mode_flows(mode)
            positive = [f for f in flows_dict.values() if f > 0]
        
        return float(np.mean(positive)) if positive else 0.0
    
    def get_statistics(self, mode: Optional[str] = None) -> Dict[str, float]:
        """
        Compute comprehensive flow statistics.
        
        Args:
            mode: Specific mode, or None for all modes
        
        Returns:
            Dictionary with keys: total, max, avg, loaded_edges
        
        Example:
            >>> car_stats = flows.get_statistics("car")
            >>> print(f"Car stats: {car_stats}")
            >>> overall_stats = flows.get_statistics()
        """
        return {
            "total_volume": self.total_volume(mode),
            "max_flow": self.max_flow(mode),
            "avg_flow": self.get_average_flow(mode),
            "loaded_edges": self.get_loaded_edges(mode),
        }
    
    def scale_flows(self, factor: float, mode: Optional[str] = None) -> MultiModalFlows:
        """
        Create a new MultiModalFlows with scaled values.
        
        Args:
            factor: Scaling multiplier (e.g., 1.5 for +50% growth)
            mode: Scale specific mode only, or None for all modes
        
        Returns:
            New MultiModalFlows instance with scaled values
        
        Example:
            >>> # Simulate 20% traffic growth
            >>> future_flows = flows.scale_flows(1.2)
            >>> 
            >>> # Double only car traffic
            >>> more_cars = flows.scale_flows(2.0, mode="car")
        """
        if factor < 0:
            raise ValueError(f"Scaling factor must be non-negative, got {factor}")
        
        def scale_dict(d: Dict[Edge, float]) -> Dict[Edge, float]:
            return {e: f * factor for e, f in d.items()}
        
        if mode is None:
            # Scale all modes
            return MultiModalFlows(
                car=scale_dict(self.car),
                ped=scale_dict(self.ped),
                transit=scale_dict(self.transit),
            )
        else:
            # Scale only specified mode
            mode = LEGACY_MODE_NAMES.get(mode, mode)
            
            if mode == "car":
                return MultiModalFlows(
                    car=scale_dict(self.car),
                    ped=self.ped.copy(),
                    transit=self.transit.copy(),
                )
            elif mode == "ped":
                return MultiModalFlows(
                    car=self.car.copy(),
                    ped=scale_dict(self.ped),
                    transit=self.transit.copy(),
                )
            elif mode == "transit":
                return MultiModalFlows(
                    car=self.car.copy(),
                    ped=self.ped.copy(),
                    transit=scale_dict(self.transit),
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
    
    def filter_edges(self, edges_to_keep: set[Edge]) -> MultiModalFlows:
        """
        Create new MultiModalFlows with only specified edges.
        
        Args:
            edges_to_keep: Set of edges to retain
        
        Returns:
            New MultiModalFlows with filtered edges
        
        Example:
            >>> # Keep only high-volume edges
            >>> total = flows.total()
            >>> busy_edges = {e for e, f in total.items() if f > 100}
            >>> filtered = flows.filter_edges(busy_edges)
        """
        def filter_dict(d: Dict[Edge, float]) -> Dict[Edge, float]:
            return {e: f for e, f in d.items() if e in edges_to_keep}
        
        return MultiModalFlows(
            car=filter_dict(self.car),
            ped=filter_dict(self.ped),
            transit=filter_dict(self.transit),
        )
    
    def add_flows(self, other: MultiModalFlows) -> MultiModalFlows:
        """
        Add flows from another MultiModalFlows instance.
        
        Args:
            other: Another MultiModalFlows to add
        
        Returns:
            New MultiModalFlows with combined flows
        
        Example:
            >>> # Combine AM and PM peak flows
            >>> daily_flows = am_flows.add_flows(pm_flows)
        """
        def combine_dicts(d1: Dict[Edge, float], d2: Dict[Edge, float]) -> Dict[Edge, float]:
            result = d1.copy()
            for edge, flow in d2.items():
                result[edge] = result.get(edge, 0.0) + flow
            return result
        
        return MultiModalFlows(
            car=combine_dicts(self.car, other.car),
            ped=combine_dicts(self.ped, other.ped),
            transit=combine_dicts(self.transit, other.transit),
        )
    
    def get_congested_edges(
        self,
        capacity_dict: Dict[Edge, float],
        threshold: float = 0.8,
        mode: Optional[str] = None,
    ) -> List[Tuple[Edge, float, float]]:
        """
        Identify edges operating near/above capacity.
        
        Args:
            capacity_dict: Dictionary mapping edges to capacities
            threshold: Utilization threshold (0 to 1)
            mode: Specific mode to check, or None for total flow
        
        Returns:
            List of (edge, flow, capacity) tuples where flow/capacity >= threshold,
            sorted by utilization ratio (descending)
        
        Example:
            >>> capacities = {(0, 1): 200, (1, 2): 100}
            >>> congested = flows.get_congested_edges(capacities, threshold=0.8)
            >>> for edge, flow, cap in congested:
            ...     print(f"Edge {edge}: {100*flow/cap:.0f}% capacity")
        """
        if mode is None:
            flow_dict = self.total()
        else:
            flow_dict = self.get_mode_flows(mode)
        
        congested = []
        
        for edge, capacity in capacity_dict.items():
            flow = flow_dict.get(edge, 0.0)
            
            if capacity > 0 and flow / capacity >= threshold:
                congested.append((edge, flow, capacity))
        
        # Sort by utilization ratio (highest first)
        congested.sort(key=lambda x: x[1] / x[2], reverse=True)
        
        return congested
    
    def summary(self) -> str:
        """
        Generate human-readable summary of flows.
        
        Returns:
            Multi-line string with flow statistics
        
        Example:
            >>> print(flows.summary())
            Multi-Modal Flow Summary:
            Car: 500.0 trips (62.5%)
            Pedestrian: 200.0 trips (25.0%)
            Transit: 100.0 trips (12.5%)
            Total: 800.0 trips
        """
        shares = self.get_all_mode_shares()
        
        lines = ["Multi-Modal Flow Summary:"]
        lines.append(f"  Car: {self.total_volume('car'):.1f} trips ({shares['car']:.1%})")
        lines.append(f"  Pedestrian: {self.total_volume('ped'):.1f} trips ({shares['ped']:.1%})")
        lines.append(f"  Transit: {self.total_volume('transit'):.1f} trips ({shares['transit']:.1%})")
        lines.append(f"  Total: {self.total_volume():.1f} trips")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MultiModalFlows("
            f"car={len(self.car)} edges, "
            f"ped={len(self.ped)} edges, "
            f"transit={len(self.transit)} edges, "
            f"total={self.total_volume():.0f} trips)"
        )


def create_empty_flows(edges: List[Edge]) -> MultiModalFlows:
    """
    Create MultiModalFlows with zero flows on all edges.
    
    Args:
        edges: List of edges to initialize
    
    Returns:
        MultiModalFlows with all flows set to 0.0
    
    Example:
        >>> edges = [(0, 1), (1, 2), (2, 3)]
        >>> flows = create_empty_flows(edges)
    """
    zero_flows = {e: 0.0 for e in edges}
    
    return MultiModalFlows(
        car=zero_flows.copy(),
        ped=zero_flows.copy(),
        transit=zero_flows.copy(),
    )


def aggregate_flows(flow_list: List[MultiModalFlows]) -> MultiModalFlows:
    """
    Aggregate multiple MultiModalFlows instances.
    
    Args:
        flow_list: List of MultiModalFlows to combine
    
    Returns:
        Single MultiModalFlows with summed values
    
    Example:
        >>> # Aggregate hourly flows into daily total
        >>> hourly_flows = [simulate_hour(h) for h in range(24)]
        >>> daily_total = aggregate_flows(hourly_flows)
    """
    if not flow_list:
        return MultiModalFlows(car={}, ped={}, transit={})
    
    result = flow_list[0]
    for flows in flow_list[1:]:
        result = result.add_flows(flows)
    
    return result