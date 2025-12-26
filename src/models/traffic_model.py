from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable, Optional

import networkx as nx
import numpy as np
import random

from models.city_grid import CityGrid

Edge = Tuple[Hashable, Hashable]


@dataclass
class TrafficModel:
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

    rng_seed: int | None = None
    trip_var_std: float = 0.02

    # Per-mode edge flows (all edges exist as keys; most will be 0.0)
    car_flows: Dict[Edge, float] = field(default_factory=dict)
    ped_flows: Dict[Edge, float] = field(default_factory=dict)
    transit_flows: Dict[Edge, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)
        # Use a seeded NumPy RNG too (avoid global np.random)
        self.nprng = np.random.default_rng(self.rng_seed)
        self.reset_flows()

    def reset_flows(self) -> None:
        G = self.grid.graph
        self.car_flows = {e: 0.0 for e in G.edges}
        self.ped_flows = {e: 0.0 for e in G.edges}
        self.transit_flows = {e: 0.0 for e in G.edges}

    def run_multimodal_assignment(self, *, deterministic_demand: bool = True) -> Dict[str, Dict[Edge, float]]:
        """
        Multi-modal assignment with realistic reachability constraints:
          - car routes only on edges whose data["modes"] contains "car"
          - ped routes only on edges whose data["modes"] contains "ped" (sidewalk network)
          - transit routes only on edges whose data["modes"] contains "transit" (transit corridors)

        Also: each mode chooses a destination that is reachable on that mode's network.
        Returns:
          {"car": {edge: flow}, "ped": {...}, "transit": {...}}
        """
        self.reset_flows()
        G = self.grid.graph

        trips_out = self.compute_trip_generations()
        dest_weights = self.compute_destination_weights()

        # Build mode-specific subgraphs (allowed edges only)
        G_car = self._subgraph_for_mode("car")
        G_ped = self._subgraph_for_mode("ped")
        G_transit = self._subgraph_for_mode("transit")

        for origin, trips in trips_out.items():
            if trips <= 0:
                continue

            # Split trips across modes
            t_car, t_ped, t_transit = self._split_modes(trips, deterministic=deterministic_demand)

            # Car
            if t_car > 0:
                dest = self._sample_reachable_destination(G_car, origin, dest_weights)
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

            # Pedestrians (use walk_time if present)
            if t_ped > 0:
                dest = self._sample_reachable_destination(G_ped, origin, dest_weights)
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

            # Transit (use transit_time if present)
            if t_transit > 0:
                dest = self._sample_reachable_destination(G_transit, origin, dest_weights)
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

        return {"car": self.car_flows, "ped": self.ped_flows, "transit": self.transit_flows}

    # ----------------------------
    # Helpers
    # ----------------------------
    def _split_modes(self, trips: float, deterministic: bool) -> tuple[float, float, float]:
        shares = np.array([self.car_share, self.ped_share, self.transit_share], dtype=float)
        s = float(shares.sum())
        if s <= 0:
            shares = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            shares /= s

        if deterministic:
            return float(trips * shares[0]), float(trips * shares[1]), float(trips * shares[2])

        total_int = int(round(trips))
        if total_int <= 0:
            return 0.0, 0.0, 0.0

        counts = self.nprng.multinomial(total_int, shares).astype(float)
        return float(counts[0]), float(counts[1]), float(counts[2])

    def _subgraph_for_mode(self, mode: str) -> nx.Graph:
        """
        Return a subgraph containing only edges that allow `mode`.
        NOTE: CityGrid now defaults edge["modes"] = {"car"}, so default here should match.
        """
        G = self.grid.graph
        edges = []
        for u, v, data in G.edges(data=True):
            modes = data.get("modes", {"car"})
            if mode in modes:
                edges.append((u, v))
        return G.edge_subgraph(edges).copy()

    def _sample_reachable_destination(
        self,
        Gm: nx.Graph,
        origin: Hashable,
        dest_weights: Dict[Hashable, float],
    ) -> Optional[Hashable]:
        """
        Pick a destination that is reachable from origin in the mode subgraph.
        Weighted by dest_weights.
        """
        if origin not in Gm:
            return None

        # Undirected graph: use connected component
        try:
            component = nx.node_connected_component(Gm, origin)
        except Exception:
            return None

        candidates = [n for n in component if n != origin]
        if not candidates:
            return None

        weights = [float(dest_weights.get(n, 1.0)) for n in candidates]
        if sum(weights) <= 0:
            weights = [1.0] * len(candidates)

        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _assign_on_graph(
        self,
        Gm: nx.Graph,
        origin: Hashable,
        dest: Hashable,
        trips: float,
        flow_dict: Dict[Edge, float],
        weight_attr: str,
        fallback_weight_attr: Optional[str],
    ) -> None:
        """
        Route trips from origin to dest on subgraph Gm and add to flow_dict.

        weight_attr: preferred edge attribute for routing
        fallback_weight_attr: used if weight_attr is missing on edges (optional)
        """
        if origin not in Gm or dest not in Gm:
            return

        # Decide which weight to use
        use_weight = weight_attr
        if fallback_weight_attr is not None:
            # If any edge lacks the preferred weight, fallback (simple heuristic)
            for _, _, data in Gm.edges(data=True):
                if use_weight not in data:
                    use_weight = fallback_weight_attr
                    break

        try:
            path = nx.shortest_path(Gm, origin, dest, weight=use_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return

        for u, v in zip(path[:-1], path[1:]):
            e = (u, v)
            if e not in flow_dict and (v, u) in flow_dict:
                e = (v, u)
            if e in flow_dict:
                flow_dict[e] += float(trips)

    def compute_trip_generations(self) -> Dict[Hashable, float]:
        trips_out: Dict[Hashable, float] = {}
        for node, data in self.grid.graph.nodes(data=True):
            pop = float(data.get("population", 0.0))
            base = pop * float(self.trips_per_person)
            noise = self.rng.gauss(1.0, float(self.trip_var_std))
            trips_out[node] = max(base * noise, 0.0)
        return trips_out

    def compute_destination_weights(self) -> Dict[Hashable, float]:
        weights: Dict[Hashable, float] = {}
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
