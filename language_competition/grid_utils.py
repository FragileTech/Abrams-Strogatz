"""Module with common utilities for manipulating population grids."""
from typing import Tuple

import numpy as np


def periodic_boundary(index: Tuple[int, int], lattice_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Apply periodic boundary conditions.

    We consider a regular lattice with periodic boundary conditions.
    per_bon function is used to apply this condition to the selected
    node. If the node is located outside the boundaries, the neighbors
    are selected applying usual nearest neighbor conditions. If the
    node is located at the boundary, the function will return a tuple
    where the neighbors are selected following periodic boundary
    conditions (in this way, the regular lattice becomes a torus).

    Args:
        index: 2D-tuple. Represents the selected node.
        lattice_shape: 2D-tupple. Shape of the lattice.
    Returns:
        The functions returns a tuple where periodic boundary conditions
        are applied if needed.

    Examples:
    If the lattice is a 5x5 array, a node located at position (6, 5) will
    result:

    >>> periodic_boundary((6, 5), (5, 5))
    (1,0)
    """
    return tuple((i % s for i, s in zip(index, lattice_shape)))


def sample_random_cell(grid: np.ndarray) -> Tuple[int, int]:
    """Return the index or a grid cell sampled at random."""
    width, height = grid.shape
    i, j = np.random.randint(width), np.random.randint(height)
    return i, j


def get_neighbor_languages(ix: Tuple[int, int], grid: np.ndarray) -> np.ndarray:
    """Return the cell values (spoken languages) of the neighbors of the target cell."""
    i, j = ix
    neighbs_ix = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))  # Indexes of cell neighbors
    return np.array([grid[periodic_boundary(ix, grid.shape)] for ix in neighbs_ix])
