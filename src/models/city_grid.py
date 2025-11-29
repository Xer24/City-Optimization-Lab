"""City grid representation.

Defines the spatial layout of the city, cell-level attributes, and helper
methods for interacting with the grid.
"""
from typing import Tuple, Iterable, Optional
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class CityGrid:
    def __init__(self, 
    width: int = 20, 
    height: int = 20,
    spacing: float = 1 ,
    diagonal: bool = False,
    *,
    seed: Optional[int] = None,
    edge_keep: float = 0.9,#probability that edge exists
    diag_keep: Optional[float] = None,
    population_range: Tuple[int, int] = (0,500), #weights
    density_range: Tuple[float, float] = (0.1, 1.0),
    patches_per_zone: int = 3): #blocks per zone
        self.width = width
        self.height = height
        self.spacing = spacing
        self.diagonal = diagonal #determines if there will be diagonal connections
        
        #RNG controls
        self.edge_keep = edge_keep
        self.diag_keep = edge_keep if diag_keep is None else diag_keep
        self.population_range = population_range
        self.density_range = density_range
        self.patches_per_zone = patches_per_zone
        self.rng = random.Random(seed)

        #Build Graph
        G = self._build_grid_graph()
        G = self._remove_isolated_nodes(G)
        G = self.keep_component(G)
        self.graph: nx.Graph = G
        self.init_block_zoning()
        self._init_random_node_attributes()
        self._init_random_edge_attributes()

    def __repr__(self) -> str:
        return f"CityGrid(width={self.width}, height={self.height}, diagonal = {self.diagonal})"

    def visualize(self, ax = None, show = True, edge_flows = None) -> None:
        self.plot_city(ax = ax, show = show, edge_flows = edge_flows)
#Helpers

    def _build_grid_graph(self) -> nx.Graph: #build the underlying grind
        # Nodes are (row, col) = (y,x)
        #Edges have attribute road which is orthogonal

        G = nx.Graph()

        #adding nodes
        for r in range(self.height):
            for c in range(self.width):
                node_id = (r,c)
                x = c * self.spacing
                y = r* self.spacing

                G.add_node(node_id,
                row = r,
                col = c,
                pos =(x, y),
                )
        #add edges
        for r in range(self.height):
            for c in range(self.width):
                node = (r,c)

            #vertvical
                if r+1 < self.height:
                    if self.rng.random() < self.edge_keep:
                        G.add_edge(node, (r+1,c), kind = "road")
                
                if c+1 < self.width:
                    if self.rng.random() < self.edge_keep:
                        G.add_edge(node, (r, c+1), kind = "road")
                if self.diagonal:
                    if (r+1 < self.height) and (c+1 < self.width):
                        if self.rng.random() < self.diag_keep:
                            G.add_edge(node, (r+1, c+1), kind = "diag")
                    if (r+1 < self.height) and (c-1 >= 0):
                        if self.rng.random() < self.diag_keep:
                            G.add_edge(node, (r+1, c-1), kind = "diag")
        return G
    
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

        avg_patch_size = max(1, total_nodes // (len(self.zones) * self.patches_per_zone))

        unassigned = set(nodes)
        zoning = {}

        # --- helper ---
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

        # --- create patches ---
        for zone in self.zones:
            for _ in range(self.patches_per_zone):
                if not unassigned:
                    break

                seed = self.rng.choice(list(unassigned))
                size_factor = self.rng.uniform(0.7, 1.3)
                target = max(1, int(avg_patch_size * size_factor))

                grow_patch(seed, zone, target)

        # --- fill leftover nodes ---
        while unassigned:
            node = unassigned.pop()
            neighbor_zones = {
                zoning[n]
                for n in self.graph.neighbors(node)
                if n in zoning
            }
            if neighbor_zones:
                zone = self.rng.choice(list(neighbor_zones))
            else:
                zone = self.rng.choice(self.zones)

            zoning[node] = zone

        # --- write zoning + colors ---
        for node, zone in zoning.items():
            self.graph.nodes[node]["zoning"] = zone
            self.graph.nodes[node]["color"] = self.zone_colors[zone]

    def _init_random_node_attributes(self) -> None:
        #Attach random attributes to nodes -> Population, zoning, density
        pop_min, pop_max = self.population_range
        dens_min, dens_max = self.density_range

        for node in self.graph.nodes:
            population = self.rng.randint(pop_min, pop_max)
            density = self.rng.uniform(dens_min, dens_max)

            #baseline energy deamnd using population and density
            baseline_energy = population * density * self.rng.uniform(0.5,1.5)

            self.graph.nodes[node]["population"] = population #setting attributes
            self.graph.nodes[node]["density"] = density
            self.graph.nodes[node]["baseline_energy"] = baseline_energy

    def _init_random_edge_attributes(self) -> None: #arrow is return type annotation
        #attach random weights to edges: Travel_time and capacity
        for u,v in self.graph.edges: #u,v are nodes
            is_diag = self.graph.edges[u,v].get("kind") == "diag"
            base_time = 1.0 if not is_diag else 1.4 #sqrt(2) about 
            #adding noise 
            travel_time = base_time * self.rng.uniform(0.8,1.5)
            capacity = self.rng.randint(50,300)
            self.graph.edges[u,v]["travel_time"] = travel_time
            self.graph.edges[u,v]["capacity"] = capacity
    

    # Ease of life methods

    def node_position(self, row: int, col:int) -> Tuple[float, float]:
        #return (x,y) coord of given cell
        return self.graph.nodes[(row,col)]["pos"]
    def neighbors(self, row: int, col: int):
        # return cell neighbors
        return list(self.graph.neighbors((row,col)))

    def _remove_isolated_nodes(self, G: nx.Graph) -> nx.Graph:
        isolated = [n for n, deg in G.degree() if deg == 0]
        G.remove_nodes_from(isolated)
        return G

    def keep_component(self,G:nx.Graph) -> nx.Graph:
        # just incase we run into condition where graph has no roads, still kep one node
        if G.number_of_edges() == 0:
            return G
        components = list(nx.connected_components(G))
        if not components:
            return G
        largest = max(components, key = len)
        G_largest = G.subgraph(largest).copy()
        return G_largest
# Legit just make graph look prettier method
    def plot_city(self, ax = None, show = True, figsize = (8,8), edge_flows = None) -> None:
        G = self.graph
        pos = nx.get_node_attributes(G, "pos")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            created_fig = True
        
        
        
        bg = "#4A6DE5"
        ax.set_facecolor(bg)
        if created_fig:
            fig.patch.set_facecolor(bg)


        #remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

        road_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("kind") == "road"]
        diag_edges = [(u, v) for u, v in G.edges if G.edges[u, v].get("kind") == "diag"]

        if edge_flows is None or not road_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax = ax,
                edgelist = road_edges if road_edges else None,
                width = 3.0,
                edge_color = "#B8C6F0",
                alpha = 0.95
            )
        else:
            #build flow array aligned w road_edges
            flows = []
            for (u,v) in road_edges:
                f = edge_flows.get((u, v), edge_flows.get((v, u), 0.0))
                flows.append(f)
            flows = np.array(flows, dtype=float)
            
            max_flow = float(flows.max()) if flows.size > 0 else 0.0

            if max_flow <= 0:
                nx.draw_networkx_edges(
                G,
                pos,
                ax = ax,
                edgelist = road_edges if road_edges else None,
                width = 3.0,
                edge_color = "#B8C6F0",
                alpha = 0.95
                )
            else:
                norm = flows/ max_flow
                widths = 1.8 + 5.0 * norm

                #draw roads colored by flow
                edge_collection = nx.draw_networkx_edges(
                    G,
                    pos,
                    ax = ax,
                    edgelist = road_edges,
                    width = widths,
                    edge_color = flows,
                    edge_cmap = plt.cm.inferno,
                    edge_vmin = 0.0,
                    edge_vmax = max_flow,
                    alpha = 0.95,

                )
                #colorbar for flows -> commented out cause its being weird rn
                #cbar = plt.colorbar(edge_collection, ax=ax, fraction=0.046, pad=0.04)
                #cbar.set_label("Traffic flow (relative)", color="white")
                #cbar.ax.yaxis.set_tick_params(color="white")
                #for label in cbar.ax.get_yticklabels():
                #    label.set_color("white")


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

        node_colors = []
        for n in G.nodes:
            zone = G.nodes[n].get("zoning","residential")
            color = self.zone_colors.get(zone, "tab:gray")
            node_colors.append(color)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax = ax,
            node_color = node_colors,
            node_size = 95,
            edgecolors = "#e4d5b7",
            linewidths = 0.35,
            alpha = 0.95

        )
        if created_fig and show:
            plt.tight_layout()
            plt.show()
            
