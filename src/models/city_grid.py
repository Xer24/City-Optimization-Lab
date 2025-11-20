"""City grid representation.

Defines the spatial layout of the city, cell-level attributes, and helper
methods for interacting with the grid.
"""

class CityGrid:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        # TODO: store per-cell attributes such as population, zoning, etc.

    def __repr__(self) -> str:
        return f"CityGrid(width={self.width}, height={self.height})"
