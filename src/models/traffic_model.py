"""
Multi-modal traffic assignment for urban transportation networks.

Implements realistic traffic flow modeling with:
- Multi-modal routing (car, pedestrian, public transit)
- Mode-specific network constraints (sidewalks, transit corridors)
- Gravity-based trip generation
- Shortest-path assignment with mode choice

Traffic Flow Process:
    1. Generate trips from each node based on population
    2. Split trips across modes using modal shares
    3. Select reachable destinations weighted by zone attractiveness
    4. Route trips on mode-specific networks
    5. Aggregate flows on edges
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable, Optional, Set, TYPE_CHECKING
import networkx as nx
import numpy as np
import random
import logging

if TYPE_CHECKING:
    from models.city_grid import CityGrid

logger = logging.getLogger(__name__)

Edge = Tuple[Hashable, Hashable]
Node = Hashable


@dataclass
class TrafficModel:
    """
    Multi-modal traffic assignment model for city grids.
    
    Simulates realistic traffic patterns by:
    - Generating trips from population centers
    - Distributing trips across car, pedestrian, and transit modes
    - Routing each mode on its allowed network (mode-specific reachability)
    - Computing edge-level flow volumes
    
    Attributes:
        grid: CityGrid with network topology and attributes
        trips_per_person: Trip generation rate (trips/person/tick)
        commercial_weight: Attractiveness multiplier for commercial zones
        industrial_weight: Attractiveness multiplier for industrial zones
        residential_weight: Attractiveness multiplier for residential zones
        car_share: Modal share for cars (auto-normalized)
        ped_share: Modal share for pedestrians (auto-normalized)
        transit_share: Modal share for public transit (auto-normalized)
        rng_seed: Random seed for reproducibility
        trip_var_std: Standard deviation for trip generation noise
        car_flows: Edge flows for car mode {edge: volume}
        ped_flows: Edge flows for pedestrian mode
        transit_flows: Edge flows for transit mode
    
    Example:
        >>> from models.city_grid import CityGrid
        >>> grid = CityGrid(width=20, height=20, seed=42)
        >>> traffic = TrafficModel(
        ...     grid,
        ...     trips_per_person=0.3,
        ...     car_share=0.6,
        ...     ped_share=0.3,
        ...     transit_share=0.1
        ... )
        >>> flows = traffic.run_multimodal_assignment()
        >>> print(f"Total car trips: {sum(flows['car'].values()):.1f}")
    """
    
    grid: CityGrid
    trips_per_person: float = 0.3
    
    # Destination attraction by zoning
    commercial_weight: float = 3.0
    industrial_weight: float = 2.0
    residential_weight: float = 1.0
    
    # Mode shares (will be normalized automatically)
    car_share: float = 0.65
    ped_share: float = 0.25
    transit_share: float = 0.10
    
    rng_seed: Optional[int] = None
    trip_var_std: float = 0.02
    
    # Per-mode edge flows (all edges exist as keys; most will be 0.0)
    car_flows: Dict[Edge, float] = field(default_factory=dict)
    ped_flows: Dict[Edge, float] = field(default_factory=dict)
    transit_flows: Dict[Edge, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize RNGs and flow dictionaries after dataclass construction.
        
        Validates parameters and sets up random number generators.
        """
        # Validation
        if self.trips_per_person < 0:
            raise ValueError(
                f"trips_per_person must be non-negative, got {self.trips_per_person}"
            )
        
        if self.trip_var_std < 0:
            raise ValueError(
                f"trip_var_std must be non-negative, got {self.trip_var_std}"
            )
        
        # Check that modal shares are non-negative
        if any(s < 0 for s in [self.car_share, self.ped_share, self.transit_share]):
            raise ValueError("Modal shares must be non-negative")
        
        total_share = self.car_share + self.ped_share + self.transit_share
        if total_share <= 0:
            raise ValueError("At least one modal share must be positive")
        
        # Initialize RNGs
        self.rng = random.Random(self.rng_seed)
        self.nprng = np.random.default_rng(self.rng_seed)
        
        # Initialize flow dictionaries
        self.reset_flows()
        
        # Log modal shares after normalization
        shares = self._normalize_shares()
        logger.info(
            f"TrafficModel initialized: car={shares[0]:.1%}, "
            f"ped={shares[1]:.1%}, transit={shares[2]:.1%}"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrafficModel(nodes={self.grid.graph.number_of_nodes()}, "
            f"trips_per_person={self.trips_per_person:.2f})"
        )
    
    def reset_flows(self) -> None:
        """
        Reset all edge flows to zero.
        
        Initializes flow dictionaries with all edges as keys.
        This ensures consistent structure across assignment runs.
        """
        G = self.grid.graph
        self.car_flows = {e: 0.0 for e in G.edges}
        self.ped_flows = {e: 0.0 for e in G.edges}
        self.transit_flows = {e: 0.0 for e in G.edges}
    
    def _normalize_shares(self) -> np.ndarray:
        """Normalize modal shares to sum to 1.0."""
        shares = np.array(
            [self.car_share, self.ped_share, self.transit_share],
            dtype=float
        )
        total = shares.sum()
        
        if total <= 0:
            logger.warning("All modal shares are zero, defaulting to 100% car")
            return np.array([1.0, 0.0, 0.0])
        
        return shares / total
    
    def run_multimodal_assignment(
        self,
        *,
        deterministic_demand: bool = True
    ) -> Dict[str, Dict[Edge, float]]:
        """
        Execute multi-modal traffic assignment.
        
        Process:
        1. Generate trips from each node based on population
        2. Split trips across modes using modal shares
        3. For each mode, route on mode-specific network:
           - Cars: edges with "car" in modes
           - Pedestrians: edges with "ped" in modes (sidewalks)
           - Transit: edges with "transit" in modes (corridors)
        4. Aggregate flows on edges
        
        Args:
            deterministic_demand: If True, use exact modal split.
                                 If False, use multinomial sampling.
        
        Returns:
            Dictionary with keys "car", "ped", "transit" mapping to
            edge flow dictionaries {edge: flow_volume}
        
        Example:
            >>> flows = traffic.run_multimodal_assignment()
            >>> car_total = sum(flows["car"].values())
            >>> ped_total = sum(flows["ped"].values())
            >>> print(f"Car: {car_total:.0f}, Pedestrian: {ped_total:.0f}")
        
        Note:
            Each mode only routes on edges where that mode is allowed.
            Destinations are sampled from the reachable set in that mode's network.
        """
        self.reset_flows()
        G = self.grid.graph
        
        # Generate trips from each node
        trips_out = self.compute_trip_generations()
        total_trips = sum(trips_out.values())
        logger.info(f"Generated {total_trips:.1f} total trips")
        
        # Compute destination attractiveness
        dest_weights = self.compute_destination_weights()
        
        # Build mode-specific subgraphs
        G_car = self._subgraph_for_mode("car")
        G_ped = self._subgraph_for_mode("ped")
        G_transit = self._subgraph_for_mode("transit")
        
        logger.debug(
            f"Mode networks: car={G_car.number_of_edges()} edges, "
            f"ped={G_ped.number_of_edges()} edges, "
            f"transit={G_transit.number_of_edges()} edges"
        )
        
        # Track assignment statistics
        assigned_trips = {"car": 0.0, "ped": 0.0, "transit": 0.0}
        failed_assignments = {"car": 0, "ped": 0, "transit": 0}
        
        # Assign trips from each origin
        for origin, trips in trips_out.items():
            if trips <= 0:
                continue
            
            # Split trips across modes
            t_car, t_ped, t_transit = self._split_modes(
                trips,
                deterministic=deterministic_demand
            )
            
            # Assign car trips
            if t_car > 0:
                dest = self._sample_reachable_destination(
                    G_car, origin, dest_weights
                )
                if dest is not None:
                    self._assign_on_graph(
                        Gm=G_car,
                        origin=origin,
                        dest=dest,
                        trips=t_car,
                        flow_dict=self.car_flows,
                        weight_attr="travel_time",
                        fallback_weight_attr=None,
                    )
                    assigned_trips["car"] += t_car
                else:
                    failed_assignments["car"] += 1
            
            # Assign pedestrian trips (use walk_time if available)
            if t_ped > 0:
                dest = self._sample_reachable_destination(
                    G_ped, origin, dest_weights
                )
                if dest is not None:
                    self._assign_on_graph(
                        Gm=G_ped,
                        origin=origin,
                        dest=dest,
                        trips=t_ped,
                        flow_dict=self.ped_flows,
                        weight_attr="walk_time",
                        fallback_weight_attr="travel_time",
                    )
                    assigned_trips["ped"] += t_ped
                else:
                    failed_assignments["ped"] += 1
            
            # Assign transit trips (use transit_time if available)
            if t_transit > 0:
                dest = self._sample_reachable_destination(
                    G_transit, origin, dest_weights
                )
                if dest is not None:
                    self._assign_on_graph(
                        Gm=G_transit,
                        origin=origin,
                        dest=dest,
                        trips=t_transit,
                        flow_dict=self.transit_flows,
                        weight_attr="transit_time",
                        fallback_weight_attr="travel_time",
                    )
                    assigned_trips["transit"] += t_transit
                else:
                    failed_assignments["transit"] += 1
        
        # Log assignment summary
        logger.info(
            f"Assignment complete: car={assigned_trips['car']:.1f}, "
            f"ped={assigned_trips['ped']:.1f}, "
            f"transit={assigned_trips['transit']:.1f}"
        )
        
        if any(failed_assignments.values()):
            logger.warning(
                f"Failed assignments (unreachable): {failed_assignments}"
            )
        
        return {
            "car": self.car_flows,
            "ped": self.ped_flows,
            "transit": self.transit_flows
        }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _split_modes(
        self,
        trips: float,
        deterministic: bool
    ) -> Tuple[float, float, float]:
        """
        Split trips across modes according to modal shares.
        
        Args:
            trips: Total trips to split
            deterministic: If True, use exact proportions.
                          If False, use multinomial sampling.
        
        Returns:
            Tuple of (car_trips, ped_trips, transit_trips)
        """
        shares = self._normalize_shares()
        
        if deterministic:
            # Exact split
            return (
                float(trips * shares[0]),
                float(trips * shares[1]),
                float(trips * shares[2])
            )
        
        # Stochastic split using multinomial distribution
        total_int = int(round(trips))
        if total_int <= 0:
            return 0.0, 0.0, 0.0
        
        counts = self.nprng.multinomial(total_int, shares).astype(float)
        return float(counts[0]), float(counts[1]), float(counts[2])
    
    def _subgraph_for_mode(self, mode: str) -> nx.Graph:
        """
        Extract subgraph of edges that allow a specific mode.
        
        Args:
            mode: Transportation mode ("car", "ped", "transit")
        
        Returns:
            Subgraph containing only edges where mode is allowed
        
        Note:
            CityGrid defaults to edge["modes"] = {"car"}, so car network
            includes all edges by default unless explicitly restricted.
        """
        G = self.grid.graph
        edges = []
        
        for u, v, data in G.edges(data=True):
            modes = data.get("modes", {"car"})  # Default to car if not specified
            if mode in modes:
                edges.append((u, v))
        
        return G.edge_subgraph(edges).copy()
    
    def _sample_reachable_destination(
        self,
        Gm: nx.Graph,
        origin: Node,
        dest_weights: Dict[Node, float],
    ) -> Optional[Node]:
        """
        Sample a destination reachable from origin in mode network.
        
        Uses weighted random sampling where weights represent zone
        attractiveness (commercial zones more attractive than residential).
        
        Args:
            Gm: Mode-specific subgraph
            origin: Origin node
            dest_weights: Attractiveness weight for each node
        
        Returns:
            Sampled destination node, or None if no destinations reachable
        """
        if origin not in Gm:
            return None
        
        # Find connected component (for undirected graph)
        try:
            component = nx.node_connected_component(Gm, origin)
        except Exception as e:
            logger.debug(f"Could not find component for {origin}: {e}")
            return None
        
        # Exclude origin from candidates
        candidates = [n for n in component if n != origin]
        
        if not candidates:
            return None
        
        # Get weights for candidates
        weights = [float(dest_weights.get(n, 1.0)) for n in candidates]
        
        # Ensure positive weights
        if sum(weights) <= 0:
            weights = [1.0] * len(candidates)
        
        # Weighted random choice
        return self.rng.choices(candidates, weights=weights, k=1)[0]
    
    def _assign_on_graph(
        self,
        Gm: nx.Graph,
        origin: Node,
        dest: Node,
        trips: float,
        flow_dict: Dict[Edge, float],
        weight_attr: str,
        fallback_weight_attr: Optional[str],
    ) -> None:
        """
        Route trips from origin to destination and update flows.
        
        Uses shortest path routing on the mode-specific network.
        
        Args:
            Gm: Mode-specific subgraph
            origin: Trip origin
            dest: Trip destination
            trips: Number of trips to assign
            flow_dict: Flow dictionary to update (car_flows, ped_flows, etc.)
            weight_attr: Preferred edge attribute for routing cost
            fallback_weight_attr: Alternative attribute if preferred missing
        
        Note:
            If weight_attr is missing on any edge, falls back to
            fallback_weight_attr for the entire path computation.
        """
        if origin not in Gm or dest not in Gm:
            return
        
        # Determine which weight attribute to use
        use_weight = weight_attr
        
        if fallback_weight_attr is not None:
            # Check if preferred weight exists on all edges
            for _, _, data in Gm.edges(data=True):
                if weight_attr not in data:
                    use_weight = fallback_weight_attr
                    break
        
        # Compute shortest path
        try:
            path = nx.shortest_path(Gm, origin, dest, weight=use_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists
            return
        
        # Add flow to each edge along path
        for u, v in zip(path[:-1], path[1:]):
            edge = (u, v)
            
            # Handle undirected edges (check both orientations)
            if edge not in flow_dict and (v, u) in flow_dict:
                edge = (v, u)
            
            if edge in flow_dict:
                flow_dict[edge] += float(trips)
    
    def compute_trip_generations(self) -> Dict[Node, float]:
        """
        Compute trip productions from each node.
        
        Trips are proportional to population with stochastic noise:
            trips = population × trips_per_person × (1 + ε)
        
        where ε ~ Normal(0, trip_var_std)
        
        Returns:
            Dictionary mapping each node to its trip production
        
        Example:
            >>> trips = traffic.compute_trip_generations()
            >>> max_node = max(trips.items(), key=lambda x: x[1])
            >>> print(f"Node {max_node[0]} generates {max_node[1]:.1f} trips")
        """
        trips_out: Dict[Node, float] = {}
        
        for node, data in self.grid.graph.nodes(data=True):
            pop = float(data.get("population", 0.0))
            base = pop * self.trips_per_person
            
            # Add stochastic variation
            if self.trip_var_std > 0:
                noise = self.rng.gauss(1.0, self.trip_var_std)
            else:
                noise = 1.0
            
            trips_out[node] = max(base * noise, 0.0)
        
        return trips_out
    
    def compute_destination_weights(self) -> Dict[Node, float]:
        """
        Compute attractiveness weights for each node as destination.
        
        Weights are based on zoning:
        - Commercial zones are most attractive (default: 3.0)
        - Industrial zones moderately attractive (default: 2.0)
        - Residential zones baseline attractive (default: 1.0)
        
        Returns:
            Dictionary mapping each node to its attractiveness weight
        """
        weights: Dict[Node, float] = {}
        
        for node, data in self.grid.graph.nodes(data=True):
            zone = data.get("zoning", "residential")
            
            if zone == "commercial":
                w = self.commercial_weight
            elif zone == "industrial":
                w = self.industrial_weight
            else:
                w = self.residential_weight
            
            weights[node] = float(w)
        
        return weights
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def get_total_flows(self) -> Dict[Edge, float]:
        """
        Aggregate flows across all modes.
        
        Returns:
            Dictionary of total flow per edge (sum of all modes)
        """
        total: Dict[Edge, float] = {}
        
        for flow_dict in [self.car_flows, self.ped_flows, self.transit_flows]:
            for edge, flow in flow_dict.items():
                total[edge] = total.get(edge, 0.0) + flow
        
        return total
    
    def get_mode_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each mode.
        
        Returns:
            Dictionary with keys "car", "ped", "transit", each containing:
            - total_trips: Sum of all flows
            - max_flow: Maximum edge flow
            - avg_flow: Average edge flow (excluding zero-flow edges)
            - loaded_edges: Number of edges with positive flow
        
        Example:
            >>> stats = traffic.get_mode_statistics()
            >>> print(f"Car max flow: {stats['car']['max_flow']:.1f}")
            >>> print(f"Pedestrian avg: {stats['ped']['avg_flow']:.1f}")
        """
        result = {}
        
        for mode, flow_dict in [
            ("car", self.car_flows),
            ("ped", self.ped_flows),
            ("transit", self.transit_flows),
        ]:
            flows = np.array(list(flow_dict.values()))
            positive_flows = flows[flows > 0]
            
            result[mode] = {
                "total_trips": float(flows.sum()),
                "max_flow": float(flows.max()) if len(flows) > 0 else 0.0,
                "avg_flow": float(positive_flows.mean()) if len(positive_flows) > 0 else 0.0,
                "loaded_edges": int(np.sum(flows > 0)),
            }
        
        return result
    
    def get_congested_edges(
        self,
        mode: str = "car",
        threshold: float = 0.8,
    ) -> list[Tuple[Edge, float, float]]:
        """
        Identify edges operating near/above capacity.
        
        Args:
            mode: Mode to analyze ("car", "ped", "transit")
            threshold: Capacity utilization threshold (0 to 1)
        
        Returns:
            List of (edge, flow, capacity) tuples where flow/capacity >= threshold
        
        Example:
            >>> congested = traffic.get_congested_edges(mode="car", threshold=0.8)
            >>> for edge, flow, cap in congested:
            ...     print(f"Edge {edge}: {flow:.0f}/{cap:.0f} = {flow/cap:.1%}")
        """
        if mode not in ["car", "ped", "transit"]:
            raise ValueError(f"Unknown mode: {mode}")
        
        flow_dict = {
            "car": self.car_flows,
            "ped": self.ped_flows,
            "transit": self.transit_flows,
        }[mode]
        
        congested = []
        
        for edge, flow in flow_dict.items():
            if flow <= 0:
                continue
            
            u, v = edge
            capacity = self.grid.graph[u][v].get("capacity", float('inf'))
            
            if capacity > 0 and flow / capacity >= threshold:
                congested.append((edge, flow, capacity))
        
        # Sort by utilization ratio (descending)
        congested.sort(key=lambda x: x[1] / x[2], reverse=True)
        
        return congested
    
    def compute_vmt(self) -> Dict[str, float]:
        """
        Compute Vehicle/Person Miles Traveled (VMT/PMT) by mode.
        
        Returns:
            Dictionary with keys "car", "ped", "transit" mapping to
            total distance traveled (flow × edge_length)
        
        Note:
            Assumes unit edge lengths unless "length" attribute present.
        """
        vmt = {"car": 0.0, "ped": 0.0, "transit": 0.0}
        
        for mode, flow_dict in [
            ("car", self.car_flows),
            ("ped", self.ped_flows),
            ("transit", self.transit_flows),
        ]:
            for edge, flow in flow_dict.items():
                u, v = edge
                length = self.grid.graph[u][v].get("length", 1.0)
                vmt[mode] += flow * length
        
        return vmt