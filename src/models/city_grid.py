"""
City grid representation with spatial layout and multi-modal transportation networks.

This module defines the core spatial structure for urban simulations, including:
- Grid-based node topology with realistic zoning
- Multi-modal edge networks (car, pedestrian, transit)
- Population and energy attributes
- Advanced visualization capabilities
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Set, Hashable, Literal
import random
import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type aliases
Node = Tuple[int, int]  # (row, col)
Edge = Tuple[Hashable, Hashable]
ZoneType = Literal["residential", "commercial", "industrial"]
TransportMode = Literal["car", "ped", "transit"]


@dataclass
class GridConfig:
    """Configuration container for city grid initialization."""
    
    width: int = 20
    height: int = 20
    spacing: float = 1.0
    diagonal: bool = False
    seed: Optional[int] = None
    edge_keep: float = 0.9
    diag_keep: Optional[float] = None
    population_range: Tuple[int, int] = (0, 500)
    density_range: Tuple[float, float] = (0.1, 1.0)
    clusters_per_zone: int = 3
    sidewalk_prob: float = 0.55
    num_transit_lines: int = 4
    transit_line_length: int = 18


class CityGrid:
    """
    Spatial grid representation of an urban area with multi-modal networks.
    
    Creates a graph-based city model with:
    - Nodes representing city blocks with zoning, population, and density
    - Edges representing streets with mode-specific accessibility
    - Realistic spatial clustering of commercial, residential, and industrial zones
    - Multi-modal transportation networks (cars, pedestrians, public transit)
    
    Attributes:
        width: Number of columns in the grid
        height: Number of rows in the grid
        spacing: Physical distance between adjacent grid cells
        diagonal: Whether to include diagonal connections
        graph: NetworkX graph representing the city structure
        zones: Tuple of available zoning types
        zone_colors: Color mapping for visualization
    
    Example:
        >>> grid = CityGrid(width=20, height=20, seed=42)
        >>> print(f"Grid has {grid.graph.number_of_nodes()} nodes")
        >>> print(f"Grid has {grid.graph.number_of_edges()} edges")
        >>> grid.visualize()
    """
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        spacing: float = 1.0,
        diagonal: bool = False,
        *,
        seed: Optional[int] = None,
        edge_keep: float = 0.9,
        diag_keep: Optional[float] = None,
        population_range: Tuple[int, int] = (0, 500),
        density_range: Tuple[float, float] = (0.1, 1.0),
        clusters_per_zone: int = 3,
    ):
        """
        Initialize city grid with specified topology and attributes.
        
        Args:
            width: Grid width (number of columns)
            height: Grid height (number of rows)
            spacing: Distance between adjacent nodes
            diagonal: Include diagonal edges if True
            seed: Random seed for reproducibility
            edge_keep: Probability of keeping each orthogonal edge [0, 1]
            diag_keep: Probability of keeping diagonal edges (defaults to edge_keep)
            population_range: Min/max population per node
            density_range: Min/max building density per node
            clusters_per_zone: Number of spatial clusters per zone type
        
        Raises:
            ValueError: If dimensions are invalid or probabilities out of range
        """
        # Validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive: {width}×{height}")
        
        if not 0 <= edge_keep <= 1:
            raise ValueError(f"edge_keep must be in [0, 1], got {edge_keep}")
        
        if population_range[0] < 0 or population_range[1] < population_range[0]:
            raise ValueError(f"Invalid population_range: {population_range}")
        
        if density_range[0] < 0 or density_range[1] < density_range[0]:
            raise ValueError(f"Invalid density_range: {density_range}")
        
        # Store configuration
        self.width = width
        self.height = height
        self.spacing = float(spacing)
        self.diagonal = diagonal
        self.edge_keep = edge_keep
        self.diag_keep = edge_keep if diag_keep is None else diag_keep
        self.population_range = population_range
        self.density_range = density_range
        self.clusters_per_zone = clusters_per_zone
        self.rng = random.Random(seed)
        
        # Build graph structure
        logger.info(f"Building {width}×{height} city grid (seed={seed})")
        G = self._build_grid_graph()
        G = self._remove_isolated_nodes(G)
        G = self._keep_largest_component(G)
        self.graph: nx.Graph = G
        
        logger.info(
            f"Grid initialized: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        # Initialize attributes
        self._init_zoning()
        self._init_node_attributes()
        self._init_edge_attributes()
        
        # Add multi-modal networks
        self.add_sidewalks(sidewalk_prob=0.55)
        self.add_random_transit_lines(num_lines=4, line_len=18)
        
        logger.info("City grid construction complete")

    def __repr__(self) -> str:
        """String representation of grid."""
        return (
            f"CityGrid(width={self.width}, height={self.height}, "
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
    
    def __str__(self) -> str:
        """Human-readable description."""
        return (
            f"{self.width}×{self.height} city grid with "
            f"{self.graph.number_of_nodes()} blocks"
        )

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    
    def _build_grid_graph(self) -> nx.Graph:
        """
        Build the underlying grid graph with random edge removal.
        
        Creates a lattice structure where:
        - Nodes are (row, col) tuples
        - Orthogonal edges are probabilistically retained
        - Diagonal edges included if self.diagonal is True
        
        Returns:
            NetworkX graph with 'pos', 'row', 'col' node attributes
            and 'kind' edge attribute ('road' or 'diag')
        """
        G = nx.Graph()
        
        # Add all nodes
        for row in range(self.height):
            for col in range(self.width):
                node_id = (row, col)
                x = col * self.spacing
                y = row * self.spacing
                G.add_node(node_id, row=row, col=col, pos=(x, y))
        
        # Add edges with probabilistic retention
        for row in range(self.height):
            for col in range(self.width):
                node = (row, col)
                
                # Vertical edge (down)
                if row + 1 < self.height and self.rng.random() < self.edge_keep:
                    G.add_edge(node, (row + 1, col), kind="road")
                
                # Horizontal edge (right)
                if col + 1 < self.width and self.rng.random() < self.edge_keep:
                    G.add_edge(node, (row, col + 1), kind="road")
                
                # Diagonal edges (if enabled)
                if self.diagonal:
                    # Southeast
                    if (row + 1 < self.height and col + 1 < self.width and
                        self.rng.random() < self.diag_keep):
                        G.add_edge(node, (row + 1, col + 1), kind="diag")
                    
                    # Southwest
                    if (row + 1 < self.height and col - 1 >= 0 and
                        self.rng.random() < self.diag_keep):
                        G.add_edge(node, (row + 1, col - 1), kind="diag")
        
        return G
    
    def _remove_isolated_nodes(self, G: nx.Graph) -> nx.Graph:
        """Remove nodes with degree 0."""
        isolated = [n for n, deg in G.degree() if deg == 0]
        if isolated:
            logger.debug(f"Removing {len(isolated)} isolated nodes")
            G.remove_nodes_from(isolated)
        return G
    
    def _keep_largest_component(self, G: nx.Graph) -> nx.Graph:
        """
        Keep only the largest connected component.
        
        Ensures the grid is fully connected for routing purposes.
        """
        if G.number_of_edges() == 0:
            logger.warning("Graph has no edges")
            return G
        
        components = list(nx.connected_components(G))
        if not components:
            return G
        
        largest = max(components, key=len)
        
        if len(components) > 1:
            total_nodes = G.number_of_nodes()
            kept_nodes = len(largest)
            logger.info(
                f"Keeping largest component: {kept_nodes}/{total_nodes} nodes "
                f"({100*kept_nodes/total_nodes:.1f}%)"
            )
        
        return G.subgraph(largest).copy()

    # =========================================================================
    # ZONING AND ATTRIBUTES
    # =========================================================================
    
    def _init_zoning(self) -> None:
        """
        Initialize spatially-clustered zoning using region growing.
        
        Creates realistic zones where similar land uses cluster together,
        avoiding the checkerboard pattern of random assignment.
        """
        self.zones: Tuple[ZoneType, ...] = ("residential", "commercial", "industrial")
        self.zone_colors: Dict[ZoneType, str] = {
            "residential": "darkviolet",
            "commercial": "darkgreen",
            "industrial": "darkred",
        }
        
        nodes = list(self.graph.nodes())
        if not nodes:
            logger.warning("No nodes to assign zoning")
            return
        
        self.rng.shuffle(nodes)
        total_nodes = len(nodes)
        
        # Target size for each cluster
        avg_patch_size = max(
            1,
            total_nodes // (len(self.zones) * self.clusters_per_zone)
        )
        
        unassigned: Set[Node] = set(nodes)
        zoning: Dict[Node, ZoneType] = {}
        
        def grow_patch(seed_node: Node, zone_label: ZoneType, target_size: int) -> None:
            """Grow a zone patch from seed using BFS."""
            queue = [seed_node]
            zoning[seed_node] = zone_label
            unassigned.discard(seed_node)
            size = 1
            
            while queue and size < target_size and unassigned:
                current = queue.pop(0)
                neighbors = list(self.graph.neighbors(current))
                self.rng.shuffle(neighbors)  # Randomize growth direction
                
                for nbr in neighbors:
                    if nbr in unassigned:
                        zoning[nbr] = zone_label
                        unassigned.discard(nbr)
                        queue.append(nbr)
                        size += 1
                        if size >= target_size:
                            break
        
        # Grow clusters for each zone type
        for zone in self.zones:
            for _ in range(self.clusters_per_zone):
                if not unassigned:
                    break
                
                seed = self.rng.choice(list(unassigned))
                size_factor = self.rng.uniform(0.7, 1.3)
                target = max(1, int(avg_patch_size * size_factor))
                grow_patch(seed, zone, target)
        
        # Assign remaining nodes based on neighbor majority
        while unassigned:
            node = unassigned.pop()
            neighbor_zones = [
                zoning[n] for n in self.graph.neighbors(node) if n in zoning
            ]
            
            if neighbor_zones:
                # Pick most common zone among neighbors
                zoning[node] = max(set(neighbor_zones), key=neighbor_zones.count)
            else:
                # Fallback to random zone
                zoning[node] = self.rng.choice(self.zones)
        
        # Apply to graph
        for node, zone in zoning.items():
            self.graph.nodes[node]["zoning"] = zone
            self.graph.nodes[node]["color"] = self.zone_colors[zone]
        
        # Log zone distribution
        zone_counts = {z: sum(1 for v in zoning.values() if v == z) for z in self.zones}
        logger.info(f"Zoning distribution: {zone_counts}")
    
    def _init_node_attributes(self) -> None:
        """
        Initialize random node attributes: population, density, baseline energy.
        
        Attributes are drawn from specified ranges and used by energy/traffic models.
        """
        pop_min, pop_max = self.population_range
        dens_min, dens_max = self.density_range
        
        for node in self.graph.nodes:
            population = self.rng.randint(pop_min, pop_max)
            density = self.rng.uniform(dens_min, dens_max)
            
            # Baseline energy proportional to population × density
            baseline_energy = population * density * self.rng.uniform(0.5, 1.5)
            
            self.graph.nodes[node]["population"] = population
            self.graph.nodes[node]["density"] = density
            self.graph.nodes[node]["baseline_energy"] = baseline_energy
    
    def _init_edge_attributes(self) -> None:
        """
        Initialize edge attributes: travel time, capacity, allowed modes.
        
        Default configuration allows only cars; sidewalks and transit
        added separately.
        """
        for u, v in self.graph.edges:
            is_diag = self.graph.edges[u, v].get("kind") == "diag"
            
            # Diagonal edges take longer to traverse
            base_time = 1.4 if is_diag else 1.0
            travel_time = base_time * self.rng.uniform(0.8, 1.5)
            
            # Random capacity for congestion modeling
            capacity = self.rng.randint(50, 300)
            
            self.graph.edges[u, v]["travel_time"] = travel_time
            self.graph.edges[u, v]["capacity"] = capacity
            self.graph.edges[u, v]["modes"] = {"car"}  # Default: car only

    # =========================================================================
    # MULTI-MODAL NETWORK CONSTRUCTION
    # =========================================================================
    
    def add_sidewalks(self, sidewalk_prob: float = 0.55) -> None:
        """
        Add pedestrian infrastructure to a subset of road edges.
        
        Args:
            sidewalk_prob: Probability that a road edge has a sidewalk [0, 1]
        
        Note:
            Pedestrian travel time is typically 20% slower than vehicle time
            on the same edge.
        """
        if not 0 <= sidewalk_prob <= 1:
            raise ValueError(f"sidewalk_prob must be in [0,1], got {sidewalk_prob}")
        
        sidewalk_count = 0
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("kind") != "road":
                continue
            
            if self.rng.random() < sidewalk_prob:
                data.setdefault("modes", set()).add("ped")
                data["walk_time"] = 1.2 * data.get("travel_time", 1.0)
                sidewalk_count += 1
        
        logger.info(
            f"Added {sidewalk_count} sidewalk edges "
            f"({100*sidewalk_count/self.graph.number_of_edges():.1f}%)"
        )
    
    def add_random_transit_lines(
        self,
        num_lines: int = 4,
        line_len: int = 18
    ) -> None:
        """
        Create sparse transit network by marking random path corridors.
        
        Args:
            num_lines: Number of transit lines to create
            line_len: Target length of each line (in edges)
        
        Note:
            Transit travel time is typically 40% faster than car travel
            due to dedicated lanes and priority signals.
        """
        nodes = list(self.graph.nodes())
        if not nodes:
            logger.warning("No nodes available for transit lines")
            return
        
        transit_edge_count = 0
        
        for line_idx in range(num_lines):
            # Random starting point
            start = self.rng.choice(nodes)
            path = [start]
            
            # Grow path randomly
            for _ in range(line_len):
                neighbors = list(self.graph.neighbors(path[-1]))
                if not neighbors:
                    break
                
                # Avoid immediate backtracking
                candidates = [n for n in neighbors if n not in path[-2:]]
                if not candidates:
                    candidates = neighbors
                
                path.append(self.rng.choice(candidates))
            
            # Mark edges along path as transit-capable
            for a, b in zip(path[:-1], path[1:]):
                data = self.graph.edges[a, b]
                if "transit" not in data.get("modes", set()):
                    data.setdefault("modes", set()).add("transit")
                    data["transit_time"] = 0.6 * data.get("travel_time", 1.0)
                    data["transit_capacity"] = 600
                    transit_edge_count += 1
            
            logger.debug(f"Transit line {line_idx+1}: {len(path)} nodes")
        
        logger.info(
            f"Added {num_lines} transit lines covering {transit_edge_count} edges"
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def node_position(self, row: int, col: int) -> Tuple[float, float]:
        """
        Get (x, y) coordinates of a node.
        
        Args:
            row: Node row index
            col: Node column index
        
        Returns:
            Tuple of (x, y) coordinates in grid space
        
        Raises:
            KeyError: If node doesn't exist
        """
        return self.graph.nodes[(row, col)]["pos"]
    
    def neighbors(self, row: int, col: int) -> List[Node]:
        """
        Get list of neighboring nodes.
        
        Args:
            row: Node row index
            col: Node column index
        
        Returns:
            List of neighbor node IDs
        """
        return list(self.graph.neighbors((row, col)))
    
    def get_zone_nodes(self, zone: ZoneType) -> List[Node]:
        """
        Get all nodes with specified zoning.
        
        Args:
            zone: Zone type ('residential', 'commercial', 'industrial')
        
        Returns:
            List of node IDs with matching zone
        """
        return [
            n for n, data in self.graph.nodes(data=True)
            if data.get("zoning") == zone
        ]
    
    def get_mode_subgraph(self, mode: TransportMode) -> nx.Graph:
        """
        Extract subgraph of edges that allow a specific mode.
        
        Args:
            mode: Transportation mode ('car', 'ped', 'transit')
        
        Returns:
            Subgraph containing only edges where mode is allowed
        """
        edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if mode in data.get("modes", set())
        ]
        return self.graph.edge_subgraph(edges).copy()
    
    def compute_network_stats(self) -> Dict[str, float]:
        """
        Compute basic network statistics.
        
        Returns:
            Dictionary with keys:
            - n_nodes: Number of nodes
            - n_edges: Number of edges
            - avg_degree: Average node degree
            - density: Network density
            - diameter: Network diameter (if connected)
        """
        G = self.graph
        stats = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "avg_degree": sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1),
            "density": nx.density(G),
        }
        
        try:
            if nx.is_connected(G):
                stats["diameter"] = nx.diameter(G)
        except:
            pass
        
        return stats

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
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
        Main visualization entry point with automatic mode selection.
        
        Args:
            ax: Matplotlib axes (creates new figure if None)
            show: Whether to display the plot immediately
            edge_flows: Single-mode flow dictionary {edge: flow}
            flow_vmax: Max flow for color normalization (single mode)
            mode: Filter to specific mode ('car', 'ped', 'transit')
            flows_by_mode: Multi-modal flows {'car': {...}, 'ped': {...}, ...}
            flow_vmax_by_mode: Per-mode max flows {'car': max, ...}
        
        Note:
            If flows_by_mode is provided, renders multi-modal visualization
            with parallel lanes. Otherwise renders single-mode view.
        """
        if flows_by_mode is not None:
            self.plot_city_multimodal(
                ax=ax,
                show=show,
                flows_by_mode=flows_by_mode,
                flow_vmax_by_mode=flow_vmax_by_mode,
            )
        else:
            self.plot_city(
                ax=ax,
                show=show,
                edge_flows=edge_flows,
                flow_vmax=flow_vmax,
                mode=mode,
            )
    
    def plot_city(
        self,
        ax=None,
        show: bool = True,
        figsize: Tuple[int, int] = (8, 8),
        edge_flows: Optional[Dict[Edge, float]] = None,
        flow_vmax: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> None:
        """
        Single-mode city visualization.
        
        Args:
            ax: Matplotlib axes
            show: Display plot immediately
            figsize: Figure size (width, height)
            edge_flows: Flow volumes by edge
            flow_vmax: Maximum flow for color scale
            mode: Filter edges to specific mode
        """
        G = self.graph
        pos = nx.get_node_attributes(G, "pos")
        
        # Create figure if needed
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        
        # Styling
        bg_color = "#4A6DE5"
        ax.set_facecolor(bg_color)
        if created_fig:
            fig.patch.set_facecolor(bg_color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        
        # Filter edges by kind and mode
        road_edges = [
            (u, v) for u, v in G.edges
            if G.edges[u, v].get("kind") == "road"
            and (mode is None or mode in G.edges[u, v].get("modes", set()))
        ]
        
        diag_edges = [
            (u, v) for u, v in G.edges
            if G.edges[u, v].get("kind") == "diag"
            and (mode is None or mode in G.edges[u, v].get("modes", set()))
        ]
        
        # Draw roads
        if edge_flows is None or not road_edges:
            # No flow data - draw uniform edges
            if road_edges:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edgelist=road_edges,
                    width=3.0, edge_color="#B8C6F0", alpha=0.95,
                )
        else:
            # Flow-based visualization
            flows = np.array([
                edge_flows.get((u, v), edge_flows.get((v, u), 0.0))
                for u, v in road_edges
            ], dtype=float)
            
            max_flow = float(flow_vmax if flow_vmax is not None else flows.max())
            
            if max_flow > 0:
                widths = 1.8 + 5.0 * (flows / max_flow)
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edgelist=road_edges,
                    width=widths, edge_color=flows,
                    edge_cmap=plt.cm.inferno,
                    edge_vmin=0.0, edge_vmax=max_flow,
                    alpha=0.95,
                )
            else:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edgelist=road_edges,
                    width=3.0, edge_color="#B8C6F0", alpha=0.95,
                )
        
        # Draw diagonals
        if diag_edges:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=diag_edges,
                width=1.8, edge_color="#A3B4E8", alpha=0.8,
            )
        
        # Draw nodes colored by zoning
        node_colors = [
            self.zone_colors.get(G.nodes[n].get("zoning", "residential"), "tab:gray")
            for n in G.nodes
        ]
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors,
            node_size=95, edgecolors="#e4d5b7",
            linewidths=0.35, alpha=0.95,
        )
        
        if created_fig and show:
            plt.tight_layout()
            plt.show()
    
    def plot_city_multimodal(
        self,
        ax=None,
        show: bool = True,
        figsize: Tuple[int, int] = (8, 8),
        flows_by_mode: Optional[Dict[str, Dict[Edge, float]]] = None,
        flow_vmax_by_mode: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Multi-modal visualization with parallel offset lanes.
        
        Renders three parallel lanes per edge:
        - Center lane: Cars (inferno colormap)
        - Offset +: Pedestrians (viridis colormap)
        - Offset -: Transit (plasma colormap)
        
        Only draws lanes where that mode is allowed on the edge.
        
        Args:
            ax: Matplotlib axes
            show: Display immediately
            figsize: Figure dimensions
            flows_by_mode: {'car': {edge: flow}, 'ped': {...}, 'transit': {...}}
            flow_vmax_by_mode: {'car': max_flow, ...} for consistent scaling
        """
        flows_by_mode = flows_by_mode or {}
        flow_vmax_by_mode = flow_vmax_by_mode or {}
        
        G = self.graph
        pos = nx.get_node_attributes(G, "pos")
        
        # Create figure
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        
        # Styling
        bg_color = "#4A6DE5"
        ax.set_facecolor(bg_color)
        if created_fig:
            fig.patch.set_facecolor(bg_color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_aspect("equal")
        
        # Get road edges
        road_edges = [
            (u, v) for u, v in G.edges
            if G.edges[u, v].get("kind") == "road"
        ]
        
        # Lane configuration
        lane_offset = 0.18 * self.spacing
        base_width = 1.6
        flow_width = 5.0
        
        lanes = [
            ("car", 0.0),      # Center
            ("ped", +1.0),     # Offset right
            ("transit", -1.0), # Offset left
        ]
        
        colormaps = {
            "car": plt.cm.inferno,
            "ped": plt.cm.viridis,
            "transit": plt.cm.plasma,
        }
        
        # Helper functions
        def get_edge_flow(mode: str, u, v) -> float:
            """Get flow for edge in specified mode."""
            flows = flows_by_mode.get(mode, {})
            return float(flows.get((u, v), flows.get((v, u), 0.0)))
        
        def get_mode_vmax(mode: str) -> float:
            """Get max flow for mode (for normalization)."""
            if mode in flow_vmax_by_mode and flow_vmax_by_mode[mode] > 0:
                return float(flow_vmax_by_mode[mode])
            
            # Compute from current flows
            vals = []
            for u, v in road_edges:
                if mode in G.edges[u, v].get("modes", set()):
                    vals.append(get_edge_flow(mode, u, v))
            
            return float(max(vals)) if vals else 1.0
        
        vmax_by_mode = {m: get_mode_vmax(m) for m, _ in lanes}
        
        # Draw faint base network
        if road_edges:
            xs, ys = [], []
            for u, v in road_edges:
                if u in pos and v in pos:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    xs.extend([x1, x2, None])
                    ys.extend([y1, y2, None])
            ax.plot(xs, ys, color="black", alpha=0.10, linewidth=1.0, zorder=1)
        
        # Draw mode-specific lanes
        for u, v in road_edges:
            if u not in pos or v not in pos:
                continue
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Edge vector and perpendicular
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            
            if length == 0:
                continue
            
            # Perpendicular unit vector (for lane offset)
            px = -dy / length
            py = dx / length
            
            edge_modes = G.edges[u, v].get("modes", set())
            
            # Draw each mode's lane
            for mode, offset_sign in lanes:
                if mode not in edge_modes:
                    continue
                
                flow = get_edge_flow(mode, u, v)
                vmax = vmax_by_mode[mode]
                
                # Compute width and color from flow
                if vmax > 0:
                    norm_flow = np.clip(flow / vmax, 0.0, 1.0)
                    width = base_width + flow_width * norm_flow
                    color = colormaps[mode](norm_flow)
                else:
                    width = base_width
                    color = colormaps[mode](0.0)
                
                # Offset position
                offset = offset_sign * lane_offset
                ox = px * offset
                oy = py * offset
                
                # Draw lane
                ax.plot(
                    [x1 + ox, x2 + ox],
                    [y1 + oy, y2 + oy],
                    color=color,
                    linewidth=width,
                    alpha=0.95,
                    solid_capstyle="round",
                    zorder=2,
                )
        
        # Draw diagonals (faint)
        diag_edges = [
            (u, v) for u, v in G.edges
            if G.edges[u, v].get("kind") == "diag"
        ]
        for u, v in diag_edges:
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="black", linewidth=1.0, alpha=0.12, zorder=0
                )
        
        # Draw nodes
        node_colors = [
            self.zone_colors.get(G.nodes[n].get("zoning", "residential"), "tab:gray")
            for n in G.nodes
        ]
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors,
            node_size=95, edgecolors="#e4d5b7",
            linewidths=0.35, alpha=0.95,
        )
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color=colormaps["car"](0.85), lw=3, label="Cars (center)"),
            Line2D([0], [0], color=colormaps["ped"](0.85), lw=3, label="Pedestrians (+)"),
            Line2D([0], [0], color=colormaps["transit"](0.85), lw=3, label="Transit (-)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=9,
            title="Mode Lanes (Color = Flow Intensity)",
        )
        
        if created_fig and show:
            plt.tight_layout()
            plt.show()