from models.city_grid import CityGrid

def main():
    # Create a 10x10 city grid
    city = CityGrid(
        width=10,
        height=10,
        spacing=1.0,
        diagonal=False,
        seed=44,
        edge_keep=0.9,
        diag_keep=None,
        population_range=(0, 500),
        density_range=(0.1, 1.0),
    )

    # Some debug info (optional)
    print(city)
    print("Nodes:", city.graph.number_of_nodes())
    print("Edges:", city.graph.number_of_edges())

    some_node = (0, 0)
    print("Node (0,0) attrs:", city.graph.nodes[some_node])

    some_edge = next(iter(city.graph.edges))
    print("Edge", some_edge, "attrs:", city.graph.edges[some_edge])

    city.visualize()

if __name__ == "__main__":
    main()
