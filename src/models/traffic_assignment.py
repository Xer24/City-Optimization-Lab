from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import numpy as np
import networkx as nx

Edge = Tuple[Any, Any]  # (u, v)

@dataclass
class AssignmentResult:
    edge_flows: Dict[Edge, float]
    total_assigned: float


def all_or_nothing_assignment(
    G: nx.DiGraph,
    node_list: List[Any],
    OD: np.ndarray,
    weight: str = "weight",
) -> AssignmentResult:
    """
    Assigns every OD[i,j] to the single shortest path from node i to node j.
    Aggregates edge flows.

    G should be directed if you have directed roads; undirected works too.
    """
    n = len(node_list)
    if OD.shape != (n, n):
        raise ValueError(f"OD must be shape {(n,n)}. Got {OD.shape}.")

    edge_flows: Dict[Edge, float] = {}
    total = 0.0

    # Precompute shortest path trees from each origin for speed
    # If your n is big, you may want caching or limiting to zone nodes only.
    for oi, o in enumerate(node_list):
        # shortest paths from origin o to all nodes
        try:
            paths = nx.single_source_dijkstra_path(G, o, weight=weight)
        except nx.NetworkXNoPath:
            continue

        for dj, d in enumerate(node_list):
            trips = float(OD[oi, dj])
            if trips <= 0 or oi == dj:
                continue

            if d not in paths:
                continue  # unreachable

            path = paths[d]  # list of nodes
            total += trips

            # add flow to each edge along path
            for u, v in zip(path[:-1], path[1:]):
                edge_flows[(u, v)] = edge_flows.get((u, v), 0.0) + trips

    return AssignmentResult(edge_flows=edge_flows, total_assigned=total)
