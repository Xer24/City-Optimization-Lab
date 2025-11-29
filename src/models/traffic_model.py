from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable

import networkx as nx
from models.city_grid import CityGrid

Edge = Tuple[Hashable, Hashable]


@dataclass
class TrafficModel:
    grid: CityGrid
    trips_per_person: float = 0.3
    commercial_weight: float = 3.0
    industrial_weight: float = 2.0
    residential_weight: float = 1.0

    edge_flows: Dict[Edge, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        G = self.grid.graph
        self.edge_flows = {edge: 0.0 for edge in G.edges}

    def run_static_assignment(self) -> Dict[Edge, float]:
        G = self.grid.graph

        trips_out = self.compute_trip_generations()
        dest_weights = self.compute_destination_weights()

        # shortest paths
        all_paths = dict(nx.all_pairs_shortest_path(G))

        for origin, trips in trips_out.items():
            if trips <= 0:
                continue

            destinations = [n for n in G.nodes if n != origin]
            if not destinations:
                continue

            weights = [dest_weights[n] for n in destinations]
            total_w = sum(weights)

            if total_w == 0:
                weights = [1.0] * len(destinations)
                total_w = float(len(destinations))

            probs = [w / total_w for w in weights]
            dest = destinations[max(range(len(destinations)), key=lambda i: probs[i])]

            path = all_paths.get(origin, {}).get(dest)
            if path is None or len(path) < 2:
                continue

            self.add_flow_along_path(path, trips)

        return self.edge_flows
#helpers

    def compute_trip_generations(self) -> Dict[Hashable, float]:
        G = self.grid.graph
        trips_out = {}
        for node, data in G.nodes(data=True):
            pop = data.get("population", 0)
            trips_out[node] = pop * self.trips_per_person
        return trips_out

    def compute_destination_weights(self) -> Dict[Hashable, float]:
        G = self.grid.graph
        weights = {}

        for node, data in G.nodes(data=True):
            zone = data.get("zoning", "residential")

            if zone == "commercial":
                w = self.commercial_weight
            elif zone == "industrial":
                w = self.industrial_weight
            else:
                w = self.residential_weight

            weights[node] = float(w)

        return weights

    def add_flow_along_path(self, path, trips: float) -> None:
        for u, v in zip(path[:-1], path[1:]):
            edge = (u, v)
            if edge not in self.edge_flows and (v, u) in self.edge_flows:
                edge = (v, u)
            if edge in self.edge_flows:
                self.edge_flows[edge] += trips
