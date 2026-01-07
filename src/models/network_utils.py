"""
Network analysis utilities for transportation systems.

Provides graph algorithms and distance computations for routing,
accessibility analysis, and network connectivity metrics.
"""

from __future__ import annotations
from typing import List, Hashable, Optional, Tuple, Dict
import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)

Node = Hashable


def all_pairs_shortest_path_lengths(
    G: nx.Graph,
    node_list: List[Node],
    weight: str = "weight",
) -> np.ndarray:
    """
    Compute all-pairs shortest path distance matrix.
    
    Uses Dijkstra's algorithm from each source node to compute
    shortest path lengths to all other nodes in the network.
    
    Args:
        G: NetworkX graph with edge weights
        node_list: Ordered list of nodes to include in distance matrix
        weight: Edge attribute to use as distance metric (default: "weight")
    
    Returns:
        n×n numpy array where element [i,j] is the shortest path distance
        from node_list[i] to node_list[j]. Unreachable pairs have np.inf.
    
    Raises:
        ValueError: If node_list is empty or contains duplicates
        
    Example:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edge(0, 1, weight=1.0)
        >>> G.add_edge(1, 2, weight=2.0)
        >>> nodes = [0, 1, 2]
        >>> dist = all_pairs_shortest_path_lengths(G, nodes)
        >>> dist[0, 2]  # Distance from node 0 to node 2
        3.0
        >>> 
        >>> # Check connectivity
        >>> unreachable = np.sum(np.isinf(dist)) - len(nodes)  # Exclude diagonal
        >>> print(f"Unreachable pairs: {unreachable}")
    
    Note:
        Time complexity is O(n * m * log(n)) where n is number of nodes
        and m is number of edges. For dense graphs or multiple queries,
        consider using Floyd-Warshall (nx.floyd_warshall_numpy) or 
        Johnson's algorithm (nx.johnson).
    """
    # Validation
    if not node_list:
        raise ValueError("node_list cannot be empty")
    
    n = len(node_list)
    
    # Check for duplicates
    if len(set(node_list)) != n:
        raise ValueError("node_list contains duplicate nodes")
    
    # Create node index mapping
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # Check that all nodes exist in graph
    missing_nodes = [n for n in node_list if n not in G]
    if missing_nodes:
        logger.warning(
            f"{len(missing_nodes)} nodes from node_list not in graph. "
            f"They will be treated as isolated."
        )
    
    # Initialize distance matrix with infinity
    dist_matrix = np.full((n, n), np.inf, dtype=float)
    
    # Compute shortest paths from each source
    nodes_processed = 0
    for source_node in node_list:
        if source_node not in G:
            # Node not in graph - set only diagonal to 0
            source_idx = node_to_idx[source_node]
            dist_matrix[source_idx, source_idx] = 0.0
            continue
        
        try:
            # Get all shortest path lengths from this source
            lengths = nx.single_source_dijkstra_path_length(
                G, source_node, weight=weight
            )
            
            source_idx = node_to_idx[source_node]
            
            # Distance to self is zero
            dist_matrix[source_idx, source_idx] = 0.0
            
            # Fill in distances to reachable targets
            for target_node, distance in lengths.items():
                if target_node in node_to_idx:
                    target_idx = node_to_idx[target_node]
                    dist_matrix[source_idx, target_idx] = float(distance)
            
            nodes_processed += 1
            
        except (nx.NetworkXError, nx.NodeNotFound) as e:
            logger.warning(
                f"Could not compute paths from {source_node}: {e}"
            )
            # Set diagonal to 0 but leave others as inf
            source_idx = node_to_idx[source_node]
            dist_matrix[source_idx, source_idx] = 0.0
            continue
    
    logger.debug(
        f"Computed shortest paths from {nodes_processed}/{n} nodes"
    )
    
    return dist_matrix


def compute_average_path_length(
    G: nx.Graph,
    node_list: Optional[List[Node]] = None,
    weight: str = "weight",
) -> float:
    """
    Compute average shortest path length across all node pairs.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to consider (default: all nodes in G)
        weight: Edge weight attribute
    
    Returns:
        Average path length, excluding unreachable pairs and self-loops
    
    Example:
        >>> avg_len = compute_average_path_length(G)
        >>> print(f"Average path length: {avg_len:.2f}")
    """
    if node_list is None:
        node_list = list(G.nodes())
    
    if not node_list:
        return 0.0
    
    dist_matrix = all_pairs_shortest_path_lengths(G, node_list, weight=weight)
    
    # Exclude diagonal (self-loops) and infinite distances (unreachable)
    mask = ~np.eye(len(node_list), dtype=bool)  # Exclude diagonal
    finite_dists = dist_matrix[mask & np.isfinite(dist_matrix)]
    
    if len(finite_dists) == 0:
        logger.warning("No finite distances found - graph may be disconnected")
        return 0.0
    
    return float(np.mean(finite_dists))


def compute_network_diameter(
    G: nx.Graph,
    node_list: Optional[List[Node]] = None,
    weight: str = "weight",
) -> float:
    """
    Compute network diameter (longest shortest path).
    
    Args:
        G: NetworkX graph
        node_list: Nodes to consider (default: all nodes)
        weight: Edge weight attribute
    
    Returns:
        Maximum finite distance between any pair of nodes.
        Returns np.inf if graph is disconnected.
    
    Example:
        >>> diameter = compute_network_diameter(G)
        >>> if np.isfinite(diameter):
        ...     print(f"Network diameter: {diameter:.2f}")
        ... else:
        ...     print("Graph is disconnected")
    """
    if node_list is None:
        node_list = list(G.nodes())
    
    if not node_list:
        return 0.0
    
    dist_matrix = all_pairs_shortest_path_lengths(G, node_list, weight=weight)
    
    # Exclude diagonal
    mask = ~np.eye(len(node_list), dtype=bool)
    off_diagonal = dist_matrix[mask]
    
    # Get max finite distance
    finite_dists = off_diagonal[np.isfinite(off_diagonal)]
    
    if len(finite_dists) == 0:
        logger.warning("No finite distances - graph is completely disconnected")
        return np.inf
    
    return float(np.max(finite_dists))


def compute_network_connectivity(
    G: nx.Graph,
    node_list: Optional[List[Node]] = None,
) -> float:
    """
    Compute fraction of node pairs that are connected.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to consider (default: all nodes)
    
    Returns:
        Ratio of connected pairs to total possible pairs (0 to 1)
    
    Example:
        >>> connectivity = compute_network_connectivity(G)
        >>> print(f"Network connectivity: {connectivity:.1%}")
    """
    if node_list is None:
        node_list = list(G.nodes())
    
    n = len(node_list)
    if n <= 1:
        return 1.0
    
    dist_matrix = all_pairs_shortest_path_lengths(G, node_list)
    
    # Count finite off-diagonal entries (connected pairs)
    mask = ~np.eye(n, dtype=bool)  # Exclude diagonal
    finite_mask = np.isfinite(dist_matrix) & mask
    connected_pairs = np.sum(finite_mask)
    
    total_pairs = n * (n - 1)  # Directed pairs
    
    return float(connected_pairs) / float(total_pairs)


def compute_accessibility_matrix(
    G: nx.Graph,
    node_list: List[Node],
    weight: str = "weight",
    decay_function: Optional[callable] = None,
) -> np.ndarray:
    """
    Compute accessibility matrix with distance decay.
    
    Accessibility from node i to node j is inversely related to distance.
    Common in urban planning for measuring spatial access to opportunities.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to include
        weight: Edge weight attribute
        decay_function: Function that maps distance to accessibility.
                       Default: exp(-distance)
    
    Returns:
        n×n array where element [i,j] is accessibility from i to j
    
    Example:
        >>> # Gravity-style decay
        >>> def gravity_decay(d):
        ...     return 1.0 / (d + 1.0)**2
        >>> 
        >>> access = compute_accessibility_matrix(G, nodes, decay_function=gravity_decay)
        >>> total_access = access.sum(axis=1)  # Total accessibility from each node
    """
    if decay_function is None:
        # Default: exponential decay
        def decay_function(d):
            return np.exp(-d)
    
    # Get distance matrix
    dist_matrix = all_pairs_shortest_path_lengths(G, node_list, weight=weight)
    
    # Apply decay function
    # Set unreachable (inf) distances to 0 accessibility
    access_matrix = np.zeros_like(dist_matrix)
    finite_mask = np.isfinite(dist_matrix)
    
    access_matrix[finite_mask] = decay_function(dist_matrix[finite_mask])
    
    return access_matrix


def find_central_nodes(
    G: nx.Graph,
    node_list: Optional[List[Node]] = None,
    weight: str = "weight",
    top_k: int = 5,
) -> List[Tuple[Node, float]]:
    """
    Find most central nodes by closeness centrality.
    
    Closeness centrality measures how close a node is to all other nodes.
    Higher values indicate more central, accessible locations.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to consider (default: all nodes)
        weight: Edge weight attribute
        top_k: Number of top nodes to return
    
    Returns:
        List of (node, centrality) tuples, sorted by centrality (descending)
    
    Example:
        >>> central = find_central_nodes(G, top_k=3)
        >>> for node, score in central:
        ...     print(f"Node {node}: centrality = {score:.3f}")
    """
    if node_list is None:
        node_list = list(G.nodes())
    
    if not node_list:
        return []
    
    dist_matrix = all_pairs_shortest_path_lengths(G, node_list, weight=weight)
    
    # Compute closeness centrality for each node
    centralities = []
    for i, node in enumerate(node_list):
        # Sum of distances to all other reachable nodes
        distances = dist_matrix[i, :]
        finite_dists = distances[np.isfinite(distances) & (distances > 0)]
        
        if len(finite_dists) > 0:
            # Closeness = 1 / average distance
            avg_dist = np.mean(finite_dists)
            centrality = 1.0 / avg_dist if avg_dist > 0 else 0.0
        else:
            centrality = 0.0
        
        centralities.append((node, centrality))
    
    # Sort by centrality (descending) and return top k
    centralities.sort(key=lambda x: x[1], reverse=True)
    
    return centralities[:top_k]


def compute_betweenness_from_paths(
    G: nx.Graph,
    node_list: List[Node],
    weight: str = "weight",
) -> Dict[Node, float]:
    """
    Compute betweenness centrality based on shortest paths.
    
    Betweenness measures how often a node lies on shortest paths
    between other nodes. High betweenness indicates importance for
    network flow/connectivity.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to consider
        weight: Edge weight attribute
    
    Returns:
        Dictionary mapping each node to its betweenness score
    
    Example:
        >>> betweenness = compute_betweenness_from_paths(G, nodes)
        >>> bottleneck = max(betweenness.items(), key=lambda x: x[1])
        >>> print(f"Bottleneck node: {bottleneck[0]}")
    
    Note:
        This is a simplified version. For large networks, use
        nx.betweenness_centrality() which is more efficient.
    """
    # Initialize betweenness scores
    betweenness = {node: 0.0 for node in node_list}
    
    # For each pair of nodes
    for source in node_list:
        if source not in G:
            continue
        
        # Compute shortest paths from source
        try:
            paths = nx.single_source_dijkstra_path(G, source, weight=weight)
        except (nx.NetworkXError, nx.NodeNotFound):
            continue
        
        for target in node_list:
            if target not in paths or source == target:
                continue
            
            path = paths[target]
            
            # Increment betweenness for all intermediate nodes
            for node in path[1:-1]:  # Exclude source and target
                if node in betweenness:
                    betweenness[node] += 1.0
    
    # Normalize by number of pairs
    n = len(node_list)
    if n > 2:
        normalization = (n - 1) * (n - 2)  # Max possible paths through a node
        for node in betweenness:
            betweenness[node] /= normalization
    
    return betweenness


def analyze_network_structure(
    G: nx.Graph,
    node_list: Optional[List[Node]] = None,
    weight: str = "weight",
) -> Dict[str, float]:
    """
    Compute comprehensive network structure metrics.
    
    Args:
        G: NetworkX graph
        node_list: Nodes to analyze (default: all nodes)
        weight: Edge weight attribute
    
    Returns:
        Dictionary with metrics:
        - avg_path_length: Average shortest path
        - diameter: Network diameter
        - connectivity: Fraction of connected pairs
        - avg_degree: Average node degree
        - density: Network density
    
    Example:
        >>> metrics = analyze_network_structure(G)
        >>> for metric, value in metrics.items():
        ...     print(f"{metric}: {value:.3f}")
    """
    if node_list is None:
        node_list = list(G.nodes())
    
    if not node_list:
        return {
            "avg_path_length": 0.0,
            "diameter": 0.0,
            "connectivity": 0.0,
            "avg_degree": 0.0,
            "density": 0.0,
        }
    
    # Create subgraph if needed
    if set(node_list) != set(G.nodes()):
        H = G.subgraph(node_list).copy()
    else:
        H = G
    
    metrics = {
        "avg_path_length": compute_average_path_length(H, node_list, weight),
        "diameter": compute_network_diameter(H, node_list, weight),
        "connectivity": compute_network_connectivity(H, node_list),
        "avg_degree": sum(d for _, d in H.degree()) / max(len(node_list), 1),
        "density": nx.density(H),
    }
    
    return metrics