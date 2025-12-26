from __future__ import annotations
import numpy as np
import networkx as nx

def all_pairs_shortest_path_lengths(G: nx.Graph, node_list: list) -> np.ndarray:
    #return distanes matrix for node_list[i]
    n = len(node_list)
    idx = {node: i for i, node in enumerate(node_list)}
    dist = np.full((n,n), np.inf, dtype = float)

    #computer shortest path lengths
    for s in node_list:
        lengths = nx.single_source_dijkstra_path_length(G, s, weight = "weight")
        si = idx[s]
        dist[si, si] = 0.0
        for t,d in lengths.items():
            if t in idx:
                dist[si, idx[t]] = float(d)
    return dist