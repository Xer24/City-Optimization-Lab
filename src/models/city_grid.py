"""City grid representation.

Defines the spatial layout of the city, cell-level attributes, and helper
methods for interacting with the grid.
"""
from typing import Tuple
import networkx as nx

class CityGrid:
    def __init__(self, 
    width: int = 10, 
    height: int = 10,
    spacing: float = 1 ,
    diagonal: bool = False):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.diagonal = diagonal #determines if there will be diagonal connections

        self.graph: nx.Graph = self._build_grid_graph()
        # TODO: initialize per-cell attributes here

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
                    G.add_edge(node, (r+1,c), kind = "road")
                
                if c+1 < self.width:
                    G.add_edge(node, (r, c+1), kind = "road")
                if self.diagonal:
                    if (r+1 < self.height) and (c+1 < self.width):
                        G.add_edge(node, (r+1, c+1), kind = "road")
                    if (r+1 < self.height) and (c-1 >= 0):
                        G.add_edge(node, (r+1, c-1), kind = "road")
        return G
    # Ease of life methods

    def node_position(self, row: int, col:int) -> Tuple[float, float]:
        #return (x,y) coord of given cell
        return self.graph.nodes[(row,col)]["pos"]
    def neighbors(self, row: int, col: int):
        # return cell neighbors
        return list(self.graph.neighbors((row,col)))