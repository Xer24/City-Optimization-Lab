"""City grid representation.

Defines the spatial layout of the city, cell-level attributes, and helper
methods for interacting with the grid.
"""
from typing import Tuple, Iterable, Optional
import random
import networkx as nx

class CityGrid:
    def __init__(self, 
    width: int = 10, 
    height: int = 10,
    spacing: float = 1 ,
    diagonal: bool = False,
    *,
    seed: Optional[int] = None,
    edge_keep: float = 0.9,#probability that edge exists
    diag_keep: Optional[float] = None,
    population_range: Tuple[int, int] = (0.500), #weights
    density_range: Tuple[float, float] = (0.1, 1.0)):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.diagonal = diagonal #determines if there will be diagonal connections
        
        #RNG controls
        self.edge_keep = edge_keep
        self.diag_keep = edge_keep if diag_keep is None else diag_keep
        self.population_range = population_range
        self.density_range = density_range
        self.rng = random.Random(seed)

        #Build Graph
        G = self._build_grid_graph()
        G = self._remove_isolated_nodes(G)
        G = self.keep_component(G)
        self.graph: nx.Graph = G
        self._init_random_node_attributes()
        self._init_random_edge_attributes()

    def __repr__(self) -> str:
        return f"CityGrid(width={self.width}, height={self.height}, diagonal = {self.diagonal})"
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
    
    def _init_random_node_attributes(self) -> None:
        #Attach random attributes to nodes -> Population, zoning, density
        zoning_cat: Iterable[str] = (
            "residential",
            "commercial",
            "industrial",
            "mixed",
        )

        pop_min, pop_max = self.population_range
        dens_min, dens_max = self.density_range

        for node in self.graph.nodes:
            population = self.rng.randint(pop_min, pop_max)
            density = self.rng.uniform(dens_min, dens_max)
            zoning = self.rng.choice(tuple(zoning_cat))

            #baseline energy deamnd using population and density
            baseline_energy = population * density * self.rng.uniform(0.5,1.5)

            self.graph.nodes[node]["population"] = population #setting attributes
            self.graph.nodes[node]["density"] = density
            self.graph.nodes[node]["zoning"] = zoning
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
