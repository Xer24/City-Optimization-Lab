import matplotlib.pyplot as plt
import networkx as nx

from models.city_grid import CityGrid
  # adjust path if your file is named differently

def main():
    # Create a 6x4 city grid
    city = CityGrid(width=6, height=4, spacing=1.0, diagonal=False)

    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    pos = nx.get_node_attributes(city.graph, "pos")

    nx.draw(city.graph, pos, with_labels=True, node_size=200, font_size=8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

if __name__ == "__main__":
    main()
