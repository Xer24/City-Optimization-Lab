"""
City grid representation.

Defines the spatial layout of the city, cell-level attributes, and helper
methods for interacting with the grid.
"""
from typing import Tuple, Optional, Dict
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class CityGrid:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        spacing: float = 1,
        diagonal: bool = False,
        *,
        seed: Optional[int] = None,
        edge_keep: float = 0.9,  # probability that edge exists
        diag_keep: Optional[float] = None,
        population_range: Tuple[int, int] = (0, 500),
        density_range: Tuple[float, float] = (0.1, 1.0),
        clusters_per_zone: int = 3,
    ):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.diagonal = diagonal

        # RNG controls
        self.edge_keep = edge_keep
        self.diag_keep = edge_keep if diag_keep is None else diag_keep
        self.population_range = population_range
        self.density_range = density_range
        self.clusters_per_zone = clusters_per_zone
        self.rng = random.Random(seed)

        # Build Graph
        G = self._build_grid_graph()
        G = self._remove_isolated_nodes(G)
        G = self.keep_component(G)
        self.graph: nx.Graph = G

        self.init_block_zoning()
        self._init_random_node_attributes()
        self._init_random_edge_attributes()

        # Realistic mode reachability overlays
        self.add_sidewalks(sidewalk_prob=0.55)
        self.add_random_transit_lines(num_lines=4, line_len=18)

    def __repr__(self) -> str:
        return f"CityGrid(width={self.width}, height={self.height}, diagonal={self.diagonal})"

    # -------------------------
    # Public visualization entry
    # -------------------------
    def visualize(
        self,
        ax=None,
        show: bool = True,
        edge_flows=None,
        flow_vmax: Optional[float] = None,
        mode: Optional[str] = None,
        flows_by_mode: Optional[Dict] = None,
        flow_vmax_by_mode: Optional[Dict] = None,
    ) -> None:
        """
        - If flows_by_mode is provided, draw ALL modes on the SAME axes:
            cars / pedestrians / transit become 3 parallel "lanes" per road edge.
            If a mode isn't allowed on an edge, that lane is drawn faint + dashed.
        - Otherwise, fallback to the single-mode plot using edge_flows (+ optional mode filter).
        """
        if flows_by_mode is not None:
            self.plot_city_multimodal(
                ax=ax,
                show=show,
                flows_by_mode=flows_by_mode,
                flow_vmax_by_mode=flow_vmax_by_mode,
            )
            return

        self.plot_city(
            ax=ax,
            show=show,
            edge_flows=edge_flows,
            flow_vmax=flow_vmax,
            mode=mode,
        )

    # -------------------------
    # Multi-modal overlay plot
    # -------------------------
    def plot_city_multimodal(
    self,
    ax=None,
    show: bool = True,
    figsize=(8, 8),
    flows_by_mode: Optional[Dict] = None,
    flow_vmax_by_mode: Optional[Dict] = None,
) -> None:
        """
        OPTION A2 (requested):
        - Parallel offset lanes (true 3-lane look)
        - Colormap intensity per mode (car=inferno, ped=viridis, transit=plasma)
        - NO dashed styling
        - Only draw lanes on edges where that mode is allowed (edge 'modes' set)

        flows_by_mode:
        {"car": {(u,v):flow}, "ped": {...}, "transit": {...}}

        flow_vmax_by_mode (optional):
        {"car": vmax, "ped": vmax, "transit": vmax}
        """
        flows_by_mode = flows_by_mode or {}
        flow_vmax_by_mode = flow_vmax_by_mode or {}

        G = self.graph
        pos = nx.get_node_attributes(G, "pos")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Background + axis styling
        bg = "#4A6DE5"
        ax.set_facecolor(bg)
        if created_fig:
            fig.patch.set_facecolor(bg)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_aspect("equal")  # important so offsets look correct

        # Road edges only
        road_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("kind") == "road"]

        # Lane offset in *data units* (scales with your grid spacing)
        lane_offset = 0.18 * float(getattr(self, "spacing", 1.0))

        # Thickness scaling
        base_lw = 1.6   # still visible at zero flow
        extra_lw = 5.0  # flow contribution

        # Per-mode lane definition: (mode, offset sign)
        lanes = [
            ("car", 0.0),
            ("ped", +1.0),
            ("transit", -1.0),
        ]

        # Per-mode colormaps
        cmaps = {
            "car": plt.cm.inferno,
            "ped": plt.cm.viridis,
            "transit": plt.cm.plasma,
        }

        def edge_flow(mode: str, u, v) -> float:
            d = flows_by_mode.get(mode, {})
            return float(d.get((u, v), d.get((v, u), 0.0)))

        def mode_vmax(mode: str) -> float:
            # If passed in, use it (for consistent scaling across ticks)
            vmax = float(flow_vmax_by_mode.get(mode, 0.0))
            if vmax > 0:
                return vmax

            # Otherwise compute from current tick over allowed edges
            vals = []
            for (u, v) in road_edges:
                modes_allowed = G.edges[u, v].get("modes", set())
                if mode in modes_allowed:
                    vals.append(edge_flow(mode, u, v))
            return float(max(vals)) if vals else 0.0

        vmax_map = {m: mode_vmax(m) for (m, _) in lanes}

        # Optional: draw faint base road network behind everything
        # (keeps structure visible even when some modes missing)
        if road_edges:
            xs = []
            ys = []
            for (u, v) in road_edges:
                if u in pos and v in pos:
                    (x1, y1) = pos[u]
                    (x2, y2) = pos[v]
                    xs.extend([x1, x2, None])
                    ys.extend([y1, y2, None])
            ax.plot(xs, ys, color="black", alpha=0.10, linewidth=1.0, zorder=1)

        # ---- Draw mode lanes (only where allowed) ----
        for (u, v) in road_edges:
            if u not in pos or v not in pos:
                continue

            (x1, y1) = pos[u]
            (x2, y2) = pos[v]

            dx = x2 - x1
            dy = y2 - y1
            L = (dx * dx + dy * dy) ** 0.5
            if L == 0:
                continue

            # Perpendicular unit vector
            nxp = -dy / L
            nyp = dx / L

            edge_modes = G.edges[u, v].get("modes", set())

            for mode, sgn in lanes:
                # Only draw this lane if mode is allowed on this edge
                if mode not in edge_modes:
                    continue

                f = edge_flow(mode, u, v)
                vmax = float(vmax_map.get(mode, 0.0))

                # Width from flow
                if vmax > 0.0:
                    w = base_lw + extra_lw * (f / vmax)
                    t = min(max(f / vmax, 0.0), 1.0)  # normalize for colormap
                else:
                    w = base_lw
                    t = 0.0

                # Offset lane
                off = sgn * lane_offset
                ox = nxp * off
                oy = nyp * off

                # Colormap color for this flow level
                cmap = cmaps.get(mode, plt.cm.inferno)
                color = cmap(t)

                ax.plot(
                    [x1 + ox, x2 + ox],
                    [y1 + oy, y2 + oy],
                    color=color,
                    linewidth=w,
                    alpha=0.95,
                    solid_capstyle="round",
                    zorder=2,
                )

        # Diagonals (optional, faint)
        diag_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("kind") == "diag"]
        for (u, v) in diag_edges:
            if u not in pos or v not in pos:
                continue
            (x1, y1) = pos[u]
            (x2, y2) = pos[v]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.12, zorder=0)

        # Nodes colored by zoning
        node_colors = []
        for n in G.nodes:
            zone = G.nodes[n].get("zoning", "residential")
            node_colors.append(self.zone_colors.get(zone, "tab:gray"))

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=95,
            edgecolors="#e4d5b7",
            linewidths=0.35,
            alpha=0.95,
        )

        # Legend (simple, mode identity only)
        legend_handles = [
            Line2D([0], [0], color=cmaps["car"](0.85), lw=3, label="Cars (center lane)"),
            Line2D([0], [0], color=cmaps["ped"](0.85), lw=3, label="Pedestrians (offset +)"),
            Line2D([0], [0], color=cmaps["transit"](0.85), lw=3, label="Transit (offset -)"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=9,
            title="Mode lanes (color=intensity)",
        )

        if created_fig and show:
            plt.tight_layout()
            plt.show()


    # -------------------------
    # Graph construction
    # -------------------------
    def _build_grid_graph(self) -> nx.Graph:
        """
        Build the underlying grid graph.

        Nodes are (row, col) = (y, x).
        Edges have attribute 'kind' = 'road' or 'diag'.
        """
        G = nx.Graph()

        # Add nodes
        for r in range(self.height):
            for c in range(self.width):
                node_id = (r, c)
                x = c * self.spacing
                y = r * self.spacing
                G.add_node(node_id, row=r, col=c, pos=(x, y))

        # Add edges
        for r in range(self.height):
            for c in range(self.width):
                node = (r, c)

                # Vertical
                if r + 1 < self.height and self.rng.random() < self.edge_keep:
                    G.add_edge(node, (r + 1, c), kind="road")

                # Horizontal
                if c + 1 < self.width and self.rng.random() < self.edge_keep:
                    G.add_edge(node, (r, c + 1), kind="road")

                # Diagonals (optional)
                if self.diagonal:
                    if (r + 1 < self.height) and (c + 1 < self.width) and (self.rng.random() < self.diag_keep):
                        G.add_edge(node, (r + 1, c + 1), kind="diag")
                    if (r + 1 < self.height) and (c - 1 >= 0) and (self.rng.random() < self.diag_keep):
                        G.add_edge(node, (r + 1, c - 1), kind="diag")

        return G

    def _remove_isolated_nodes(self, G: nx.Graph) -> nx.Graph:
        isolated = [n for n, deg in G.degree() if deg == 0]
        G.remove_nodes_from(isolated)
        return G

    def keep_component(self, G: nx.Graph) -> nx.Graph:
        """
        Keep the largest connected component (if there are edges).
        """
        if G.number_of_edges() == 0:
            return G
        components = list(nx.connected_components(G))
        if not components:
            return G
        largest = max(components, key=len)
        return G.subgraph(largest).copy()

    # -------------------------
    # Zoning + node attrs
    # -------------------------
    def init_block_zoning(self) -> None:
        self.zones = ("residential", "commercial", "industrial")
        self.zone_colors = {
            "residential": "darkviolet",
            "commercial": "darkgreen",
            "industrial": "darkred",
        }

        nodes = list(self.graph.nodes)
        self.rng.shuffle(nodes)

        total_nodes = len(nodes)
        if total_nodes == 0:
            return

        avg_patch_size = max(1, total_nodes // (len(self.zones) * self.clusters_per_zone))

        unassigned = set(nodes)
        zoning = {}

        def grow_patch(seed_node, zone_label, target_size):
            queue = [seed_node]
            zoning[seed_node] = zone_label
            unassigned.discard(seed_node)
            size = 1

            while queue and size < target_size and unassigned:
                current = queue.pop(0)
                for nbr in self.graph.neighbors(current):
                    if nbr in unassigned:
                        zoning[nbr] = zone_label
                        unassigned.discard(nbr)
                        queue.append(nbr)
                        size += 1
                        if size >= target_size:
                            break

        for zone in self.zones:
            for _ in range(self.clusters_per_zone):
                if not unassigned:
                    break
                seed = self.rng.choice(list(unassigned))
                size_factor = self.rng.uniform(0.7, 1.3)
                target = max(1, int(avg_patch_size * size_factor))
                grow_patch(seed, zone, target)

        while unassigned:
            node = unassigned.pop()
            neighbor_zones = {zoning[n] for n in self.graph.neighbors(node) if n in zoning}
            zoning[node] = self.rng.choice(list(neighbor_zones)) if neighbor_zones else self.rng.choice(self.zones)

        # Write zoning + colors
        for node, zone in zoning.items():
            self.graph.nodes[node]["zoning"] = zone
            self.graph.nodes[node]["color"] = self.zone_colors[zone]

    def _init_random_node_attributes(self) -> None:
        pop_min, pop_max = self.population_range
        dens_min, dens_max = self.density_range

        for node in self.graph.nodes:
            population = self.rng.randint(pop_min, pop_max)
            density = self.rng.uniform(dens_min, dens_max)
            baseline_energy = population * density * self.rng.uniform(0.5, 1.5)

            self.graph.nodes[node]["population"] = population
            self.graph.nodes[node]["density"] = density
            self.graph.nodes[node]["baseline_energy"] = baseline_energy

    # -------------------------
    # Edge attrs + mode overlays
    # -------------------------
    def _init_random_edge_attributes(self) -> None:
        """
        Attach edge attributes:
          - travel_time
          - capacity
          - modes (allowed modes), default {"car"} for all edges
        """
        for u, v in self.graph.edges:
            is_diag = self.graph.edges[u, v].get("kind") == "diag"
            base_time = 1.0 if not is_diag else 1.4
            travel_time = base_time * self.rng.uniform(0.8, 1.5)
            capacity = self.rng.randint(50, 300)

            self.graph.edges[u, v]["travel_time"] = travel_time
            self.graph.edges[u, v]["capacity"] = capacity

            # Default: cars only. We'll add sidewalks/transit separately.
            self.graph.edges[u, v]["modes"] = {"car"}

    def add_sidewalks(self, sidewalk_prob: float = 0.55) -> None:
        """
        Mark a subset of road edges as walkable by pedestrians.
        sidewalk_prob controls how dense the pedestrian network is.
        """
        for u, v, data in self.graph.edges(data=True):
            if data.get("kind") != "road":
                continue
            if self.rng.random() < sidewalk_prob:
                data.setdefault("modes", set()).add("ped")
                data["walk_time"] = 1.2 * data.get("travel_time", 1.0)

    def add_random_transit_lines(self, num_lines: int = 4, line_len: int = 18) -> None:
        """
        Create a sparse transit network by marking a few corridors as transit-capable.
        """
        nodes = list(self.graph.nodes)
        if not nodes:
            return

        for _ in range(num_lines):
            start = self.rng.choice(nodes)
            path = [start]
            for _ in range(line_len):
                nbrs = list(self.graph.neighbors(path[-1]))
                if not nbrs:
                    break
                path.append(self.rng.choice(nbrs))

            for a, b in zip(path[:-1], path[1:]):
                data = self.graph.edges[a, b]
                data.setdefault("modes", set()).add("transit")
                data["transit_time"] = 0.6 * data.get("travel_time", 1.0)
                data["transit_capacity"] = 600

    # -------------------------
    # Convenience helpers
    # -------------------------
    def node_position(self, row: int, col: int) -> Tuple[float, float]:
        return self.graph.nodes[(row, col)]["pos"]

    def neighbors(self, row: int, col: int):
        return list(self.graph.neighbors((row, col)))

    # -------------------------
    # Single-mode plotting
    # -------------------------
    def plot_city(
        self,
        ax=None,
        show: bool = True,
        figsize=(8, 8),
        edge_flows=None,
        flow_vmax: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> None:
        """
        Draw the city graph with optional edge flow visualization.

        mode:
          - None: draw all road/diag edges
          - "car"/"ped"/"transit": draw only edges that allow that mode
        """
        G = self.graph
        pos = nx.get_node_attributes(G, "pos")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        bg = "#4A6DE5"
        ax.set_facecolor(bg)
        if created_fig:
            fig.patch.set_facecolor(bg)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

        # Filter edges by kind + allowed mode
        road_edges = [
            (u, v)
            for u, v in G.edges
            if G.edges[u, v].get("kind") == "road"
            and (mode is None or mode in G.edges[u, v].get("modes", set()))
        ]

        diag_edges = [
            (u, v)
            for u, v in G.edges
            if G.edges[u, v].get("kind") == "diag"
            and (mode is None or mode in G.edges[u, v].get("modes", set()))
        ]

        # Draw roads (with or without flows)
        if edge_flows is None or not road_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=road_edges if road_edges else None,
                width=3.0,
                edge_color="#B8C6F0",
                alpha=0.95,
            )
        else:
            flows = []
            for (u, v) in road_edges:
                f = edge_flows.get((u, v), edge_flows.get((v, u), 0.0))
                flows.append(f)
            flows = np.array(flows, dtype=float)

            tick_max = float(flows.max()) if flows.size > 0 else 0.0
            max_flow = float(flow_vmax) if flow_vmax is not None else tick_max

            if max_flow <= 0:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edgelist=road_edges if road_edges else None,
                    width=3.0,
                    edge_color="#B8C6F0",
                    alpha=0.95,
                )
            else:
                norm = flows / max_flow
                widths = 1.8 + 5.0 * norm

                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edgelist=road_edges,
                    width=widths,
                    edge_color=flows,
                    edge_cmap=plt.cm.inferno,
                    edge_vmin=0.0,
                    edge_vmax=max_flow,
                    alpha=0.95,
                )

        # Draw diagonals lightly
        if diag_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=diag_edges,
                width=1.8,
                edge_color="#A3B4E8",
                alpha=0.8,
                style="solid",
            )

        # Draw nodes colored by zoning (same for all modes)
        node_colors = []
        for n in G.nodes:
            zone = G.nodes[n].get("zoning", "residential")
            node_colors.append(self.zone_colors.get(zone, "tab:gray"))

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=95,
            edgecolors="#e4d5b7",
            linewidths=0.35,
            alpha=0.95,
        )

        if created_fig and show:
            plt.tight_layout()
            plt.show()
