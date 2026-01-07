"""
Traffic assignment algorithms for network flow modeling.

Implements various assignment methods for distributing origin-destination
trip demand across transportation networks:
- All-or-nothing assignment (shortest path)
- Stochastic assignment (multi-path with logit choice)
- Incremental assignment
- Method of Successive Averages (MSA)

These algorithms are fundamental building blocks for traffic simulation
and can be used standalone or within iterative equilibrium procedures.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional, TYPE_CHECKING
import numpy as np
import networkx as nx
import logging

if TYPE_CHECKING:
    pass  # Add any type-only imports here if needed

logger = logging.getLogger(__name__)

Edge = Tuple[Any, Any]  # (u, v)
Node = Any


@dataclass
class AssignmentResult:
    """
    Container for traffic assignment results.
    
    Attributes:
        edge_flows: Dictionary mapping edges to assigned flow volumes
        total_assigned: Total trips successfully assigned to the network
        unassigned_trips: Trips that couldn't be assigned (unreachable OD pairs)
        computation_time: Time taken for assignment (optional)
        num_paths_found: Number of OD pairs with valid paths
    
    Example:
        >>> result = all_or_nothing_assignment(G, nodes, OD)
        >>> print(f"Assigned {result.total_assigned:.0f} trips")
        >>> print(f"Max edge flow: {max(result.edge_flows.values()):.0f}")
    """
    
    edge_flows: Dict[Edge, float]
    total_assigned: float
    unassigned_trips: float = 0.0
    computation_time: Optional[float] = None
    num_paths_found: int = 0
    
    def get_flow(self, edge: Edge) -> float:
        """
        Get flow on a specific edge.
        
        Args:
            edge: Edge tuple (u, v)
        
        Returns:
            Flow on edge, or 0.0 if edge has no flow
        """
        return self.edge_flows.get(edge, 0.0)
    
    def max_flow(self) -> float:
        """Get maximum flow across all edges."""
        return max(self.edge_flows.values()) if self.edge_flows else 0.0
    
    def total_flow(self) -> float:
        """Get sum of all edge flows."""
        return sum(self.edge_flows.values())
    
    def loaded_edges(self) -> int:
        """Count number of edges with positive flow."""
        return sum(1 for f in self.edge_flows.values() if f > 0)
    
    def avg_flow(self) -> float:
        """Get average flow on loaded edges."""
        positive_flows = [f for f in self.edge_flows.values() if f > 0]
        return np.mean(positive_flows) if positive_flows else 0.0
    
    def assignment_rate(self) -> float:
        """
        Calculate fraction of trips successfully assigned.
        
        Returns:
            Ratio of assigned trips to total demand (0 to 1)
        """
        total_demand = self.total_assigned + self.unassigned_trips
        if total_demand <= 0:
            return 0.0
        return self.total_assigned / total_demand


def all_or_nothing_assignment(
    G: nx.Graph | nx.DiGraph,
    node_list: List[Node],
    OD: np.ndarray,
    weight: str = "weight",
) -> AssignmentResult:
    """
    All-or-nothing traffic assignment using shortest paths.
    
    Assigns all demand from each OD pair to the single shortest path,
    without considering capacity constraints or congestion. This is the
    fundamental building block for more sophisticated assignment methods.
    
    Algorithm:
        1. For each origin, compute shortest path tree
        2. For each OD pair with demand > 0:
           - Find shortest path from origin to destination
           - Add all trips to edges along that path
        3. Aggregate flows across all OD pairs
    
    Args:
        G: NetworkX graph (directed or undirected) representing network
        node_list: Ordered list of nodes matching OD matrix indices
        OD: n×n origin-destination trip matrix where OD[i,j] = trips from
            node_list[i] to node_list[j]
        weight: Edge attribute to use as routing cost (default: "weight")
    
    Returns:
        AssignmentResult with edge flows and assignment statistics
    
    Raises:
        ValueError: If OD matrix dimensions don't match node_list length
    
    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edge('A', 'B', weight=1.0)
        >>> G.add_edge('B', 'C', weight=1.0)
        >>> nodes = ['A', 'B', 'C']
        >>> OD = np.array([[0, 100, 0], [0, 0, 50], [0, 0, 0]])
        >>> result = all_or_nothing_assignment(G, nodes, OD)
        >>> print(f"Total assigned: {result.total_assigned}")
        100.0
    
    Note:
        This method produces deterministic results where all travelers
        use the shortest path. Real behavior may involve path diversity,
        which can be captured with stochastic_assignment().
    
    Time Complexity:
        O(n * (m + n*log(n))) where n = nodes, m = edges
        Uses Dijkstra's algorithm for each origin.
    """
    import time
    start_time = time.time()
    
    n = len(node_list)
    
    # Validation
    if OD.shape != (n, n):
        raise ValueError(
            f"OD matrix shape {OD.shape} doesn't match node_list length {n}"
        )
    
    if n == 0:
        logger.warning("Empty node list provided")
        return AssignmentResult(
            edge_flows={},
            total_assigned=0.0,
            unassigned_trips=0.0,
        )
    
    edge_flows: Dict[Edge, float] = {}
    total_assigned = 0.0
    unassigned_trips = 0.0
    num_paths_found = 0
    unreachable_pairs = []
    
    # Precompute shortest path trees from each origin
    logger.debug(f"Computing assignment for {n} origins")
    
    for oi, origin in enumerate(node_list):
        if origin not in G:
            logger.warning(f"Origin {origin} not in graph, skipping")
            # Count unassigned trips from this origin
            unassigned_trips += float(OD[oi, :].sum())
            continue
        
        # Compute shortest paths from this origin to all nodes
        try:
            paths = nx.single_source_dijkstra_path(G, origin, weight=weight)
        except (nx.NetworkXError, nx.NodeNotFound) as e:
            logger.warning(f"Cannot compute paths from {origin}: {e}")
            unassigned_trips += float(OD[oi, :].sum())
            continue
        
        # Assign trips to destinations
        for dj, dest in enumerate(node_list):
            trips = float(OD[oi, dj])
            
            # Skip if no demand or internal trip
            if trips <= 0 or oi == dj:
                continue
            
            # Check if destination is reachable
            if dest not in paths:
                unreachable_pairs.append((origin, dest))
                unassigned_trips += trips
                continue
            
            # Get the shortest path
            path = paths[dest]
            total_assigned += trips
            num_paths_found += 1
            
            # Add flow to each edge along the path
            for u, v in zip(path[:-1], path[1:]):
                edge = (u, v)
                edge_flows[edge] = edge_flows.get(edge, 0.0) + trips
    
    computation_time = time.time() - start_time
    
    # Log summary
    logger.info(
        f"All-or-nothing assignment complete: "
        f"{total_assigned:.1f} trips assigned to {len(edge_flows)} edges "
        f"in {computation_time:.2f}s"
    )
    
    if unreachable_pairs:
        logger.warning(
            f"Found {len(unreachable_pairs)} unreachable OD pairs, "
            f"{unassigned_trips:.1f} trips unassigned"
        )
        if len(unreachable_pairs) <= 5:
            logger.debug(f"Unreachable pairs: {unreachable_pairs}")
    
    return AssignmentResult(
        edge_flows=edge_flows,
        total_assigned=total_assigned,
        unassigned_trips=unassigned_trips,
        computation_time=computation_time,
        num_paths_found=num_paths_found,
    )


def stochastic_assignment(
    G: nx.Graph | nx.DiGraph,
    node_list: List[Node],
    OD: np.ndarray,
    weight: str = "weight",
    theta: float = 1.0,
    num_paths: int = 5,
    seed: Optional[int] = None,
) -> AssignmentResult:
    """
    Stochastic traffic assignment using logit path choice.
    
    Distributes trips across multiple paths based on path costs,
    with probability proportional to exp(-θ × cost). This better
    represents real-world route choice where travelers don't all
    use the absolute shortest path.
    
    Algorithm:
        1. For each OD pair, find k shortest paths
        2. Compute path utilities: U_k = -θ × cost_k
        3. Compute logit probabilities: P_k = exp(U_k) / Σ exp(U_i)
        4. Split trips across paths according to probabilities
    
    Args:
        G: Transportation network graph
        node_list: List of nodes matching OD matrix
        OD: Trip demand matrix
        weight: Edge cost attribute
        theta: Path choice parameter (higher = more deterministic)
                θ → 0: uniform split, θ → ∞: all-or-nothing
        num_paths: Number of alternative paths to consider per OD pair
        seed: Random seed for tie-breaking in path generation
    
    Returns:
        AssignmentResult with probabilistically distributed flows
    
    Example:
        >>> # More realistic route choice with theta=1.0
        >>> result = stochastic_assignment(G, nodes, OD, theta=1.0)
        >>> # Nearly deterministic (like all-or-nothing)
        >>> result = stochastic_assignment(G, nodes, OD, theta=10.0)
    
    Note:
        Higher theta values make assignment more deterministic.
        Typical values: 0.5-2.0 for urban networks.
    """
    import time
    start_time = time.time()
    
    n = len(node_list)
    
    if OD.shape != (n, n):
        raise ValueError(
            f"OD matrix shape {OD.shape} doesn't match node_list length {n}"
        )
    
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    
    edge_flows: Dict[Edge, float] = {}
    total_assigned = 0.0
    unassigned_trips = 0.0
    num_paths_found = 0
    
    logger.debug(f"Stochastic assignment: θ={theta}, k={num_paths}")
    
    for oi, origin in enumerate(node_list):
        for dj, dest in enumerate(node_list):
            trips = float(OD[oi, dj])
            
            if trips <= 0 or oi == dj:
                continue
            
            if origin not in G or dest not in G:
                unassigned_trips += trips
                continue
            
            # Find k shortest paths
            try:
                path_generator = nx.shortest_simple_paths(
                    G, origin, dest, weight=weight
                )
                paths = []
                for _ in range(num_paths):
                    try:
                        paths.append(next(path_generator))
                    except StopIteration:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                unassigned_trips += trips
                continue
            
            if not paths:
                unassigned_trips += trips
                continue
            
            # Compute path costs
            path_costs = []
            for path in paths:
                cost = sum(
                    G[u][v].get(weight, 1.0)
                    for u, v in zip(path[:-1], path[1:])
                )
                path_costs.append(cost)
            
            # Compute logit probabilities
            utilities = [-theta * c for c in path_costs]
            max_util = max(utilities)
            exp_utils = [np.exp(u - max_util) for u in utilities]
            sum_exp = sum(exp_utils)
            probabilities = [e / sum_exp for e in exp_utils]
            
            # Distribute trips across paths
            for path, prob in zip(paths, probabilities):
                flow = trips * prob
                total_assigned += flow
                
                for u, v in zip(path[:-1], path[1:]):
                    edge = (u, v)
                    edge_flows[edge] = edge_flows.get(edge, 0.0) + flow
            
            num_paths_found += 1
    
    computation_time = time.time() - start_time
    
    logger.info(
        f"Stochastic assignment complete: "
        f"{total_assigned:.1f} trips on {len(edge_flows)} edges "
        f"in {computation_time:.2f}s"
    )
    
    return AssignmentResult(
        edge_flows=edge_flows,
        total_assigned=total_assigned,
        unassigned_trips=unassigned_trips,
        computation_time=computation_time,
        num_paths_found=num_paths_found,
    )


def incremental_assignment(
    G: nx.Graph | nx.DiGraph,
    node_list: List[Node],
    OD: np.ndarray,
    weight: str = "weight",
    num_iterations: int = 10,
    update_weights_fn: Optional[callable] = None,
) -> AssignmentResult:
    """
    Incremental assignment with congestion feedback.
    
    Assigns demand in chunks, updating edge costs after each iteration
    to reflect increasing congestion. Provides a simple approximation
    to user equilibrium.
    
    Algorithm:
        1. Split OD matrix into n equal chunks
        2. For each chunk:
           - Perform all-or-nothing assignment
           - Add flows to cumulative total
           - Update edge weights based on congestion
        3. Return cumulative flows
    
    Args:
        G: Transportation network
        node_list: Nodes matching OD matrix
        OD: Trip demand matrix
        weight: Base edge cost attribute
        num_iterations: Number of incremental steps (e.g., 10 = 10% chunks)
        update_weights_fn: Optional function(G, flows) to update edge costs
                          based on current flows. If None, uses simple
                          linear congestion model.
    
    Returns:
        AssignmentResult with flows considering incremental congestion
    
    Example:
        >>> # Custom congestion function
        >>> def bpr_update(G, flows):
        ...     for (u, v), flow in flows.items():
        ...         capacity = G[u][v].get('capacity', 1000)
        ...         t0 = G[u][v].get('freeflow_time', 1.0)
        ...         G[u][v]['weight'] = t0 * (1 + 0.15 * (flow/capacity)**4)
        >>> 
        >>> result = incremental_assignment(
        ...     G, nodes, OD, 
        ...     num_iterations=10,
        ...     update_weights_fn=bpr_update
        ... )
    """
    import time
    start_time = time.time()
    
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")
    
    # Initialize cumulative flows
    cumulative_flows: Dict[Edge, float] = {}
    total_assigned = 0.0
    total_unassigned = 0.0
    
    # Split OD matrix into chunks
    chunk_OD = OD / num_iterations
    
    logger.info(f"Incremental assignment: {num_iterations} iterations")
    
    for iteration in range(num_iterations):
        # Assign this chunk
        result = all_or_nothing_assignment(G, node_list, chunk_OD, weight)
        
        # Add to cumulative flows
        for edge, flow in result.edge_flows.items():
            cumulative_flows[edge] = cumulative_flows.get(edge, 0.0) + flow
        
        total_assigned += result.total_assigned
        total_unassigned += result.unassigned_trips
        
        # Update weights for next iteration (except last)
        if iteration < num_iterations - 1:
            if update_weights_fn is not None:
                update_weights_fn(G, cumulative_flows)
            else:
                # Simple linear congestion model
                for edge, flow in cumulative_flows.items():
                    u, v = edge
                    if G.has_edge(u, v):
                        base_weight = G[u][v].get('base_weight', 1.0)
                        capacity = G[u][v].get('capacity', 1000.0)
                        congestion_factor = 1.0 + 0.5 * (flow / capacity)
                        G[u][v][weight] = base_weight * congestion_factor
        
        logger.debug(
            f"Iteration {iteration+1}/{num_iterations}: "
            f"{result.total_assigned:.1f} trips assigned"
        )
    
    computation_time = time.time() - start_time
    
    logger.info(
        f"Incremental assignment complete: "
        f"{total_assigned:.1f} total trips in {computation_time:.2f}s"
    )
    
    return AssignmentResult(
        edge_flows=cumulative_flows,
        total_assigned=total_assigned,
        unassigned_trips=total_unassigned,
        computation_time=computation_time,
    )


def compare_assignments(
    results: Dict[str, AssignmentResult],
    metric: str = "edge_flows",
) -> Dict[str, Any]:
    """
    Compare multiple assignment results.
    
    Args:
        results: Dictionary mapping method names to AssignmentResults
        metric: Comparison metric ("edge_flows", "statistics", "correlation")
    
    Returns:
        Dictionary with comparison data
    
    Example:
        >>> aon = all_or_nothing_assignment(G, nodes, OD)
        >>> stoch = stochastic_assignment(G, nodes, OD, theta=1.0)
        >>> comparison = compare_assignments({
        ...     "all-or-nothing": aon,
        ...     "stochastic": stoch
        ... })
        >>> print(comparison["flow_correlation"])
    """
    if metric == "statistics":
        return {
            name: {
                "total_assigned": result.total_assigned,
                "max_flow": result.max_flow(),
                "avg_flow": result.avg_flow(),
                "loaded_edges": result.loaded_edges(),
            }
            for name, result in results.items()
        }
    
    elif metric == "correlation":
        # Compute pairwise flow correlations
        from scipy.stats import pearsonr
        
        names = list(results.keys())
        n = len(names)
        corr_matrix = np.ones((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:
                    flows1 = results[name1].edge_flows
                    flows2 = results[name2].edge_flows
                    
                    # Get common edges
                    edges = set(flows1.keys()) | set(flows2.keys())
                    f1 = [flows1.get(e, 0.0) for e in edges]
                    f2 = [flows2.get(e, 0.0) for e in edges]
                    
                    if len(edges) > 1:
                        corr, _ = pearsonr(f1, f2)
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        return {
            "names": names,
            "correlation_matrix": corr_matrix,
        }
    
    else:  # edge_flows
        return {
            name: result.edge_flows
            for name, result in results.items()
        }
