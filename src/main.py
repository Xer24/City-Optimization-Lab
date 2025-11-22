import matplotlib.pyplot as plt
import networkx as nx
from models.city_grid import CityGrid
  # adjust path if your file is named differently

def main():
    # Create a 6x4 city grid
    city = CityGrid(
    width=10,
    height=10,
    spacing=1.0,
    diagonal=False,
    seed=44,
    edge_keep=0.9,
    diag_keep=None,
    population_range=(0, 500),  # MUST be a tuple
    density_range=(0.1, 1.0),   # MUST be a tuple
)


    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    some_node = (0, 0)
    print("Node (0,0) attrs:", city.graph.nodes[some_node])

    some_edge = next(iter(city.graph.edges))
    print("Edge", some_edge, "attrs:", city.graph.edges[some_edge])

    G = city.graph
    pos = nx.get_node_attributes(G, "pos")

    node_colors_attr = nx.get_node_attributes(G, "color")
    colors = [node_colors_attr.get(node, "tab:gray") for node in G.nodes]
    missing = [n for n in G.nodes if "color" not in G.nodes[n]]
    print("Nodes missing color:", missing)
    nx.draw(G, pos, node_color=colors, node_size=200, with_labels=False)
    plt.gca().set_aspect("equal", adjustable="box")  # optional but nice
    plt.show()

if __name__ == "__main__":
    main()
