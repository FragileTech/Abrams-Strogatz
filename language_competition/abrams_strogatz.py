"""Abrams-strogatz language competition model."""
from typing import Tuple

import numpy as np


def initialize_grid(shape: Tuple[int, int], prob_a: float = 0.5) -> np.ndarray:
    """
    Create the initial (height x width) array.

    Each point within the (height x width) array represents a citizen speaking
    language A or language B. The default initial condition corresponds
    to a scenario where the speakers are randomly distributed, the
    probability of each speaker being p(A) = pa and p(B) = 1 - prob_a.

    Args:
        shape: (height, width) of the initialized grid.
        prob_a: Probability that a single node within the array speaks
            language A. Defaults to 0.5.

    Returns:
        Returns a np.array(shape=(height, width)) where each node speaks either
        language A (represented by a value 1) or language B (represented
        by a value -1)

    """
    height, width = shape
    mask = np.random.rand(height, width)
    grid = np.where(mask < prob_a, 1, -1)
    return grid


def update_grid(
    grid: np.array,
    cell_ix: Tuple[int, int],
    neighbors: np.ndarray,
    status_a: float = 0.5,
    vol: float = 1.0,
) -> np.array:
    """
    Update one node of the grid containing a grid of speakers following the\
     Abrams-Strogatz language model.

    Each time this function is called, it computes the probability of change for a selected node.
    The probability to change from language A to B is:
    pAB = (1 - s) * nB ** vol
    The probability to change from language B to A is:
    pBA = s * nA ** vol

    The steps are the following:
    1) A random node is selected.
    2) The language of each neighbor surrounding the node is computed,
    counting the number of speakers.
    3) The probability of change is computed. If the probability is
    bigger than a uniformly distributed random number, the language
    of the selected node is changed.

    Args:
        grid: Array containing the language spoken by the grid.
            The values contained inside the array are 1 (lang A) and
            -1 (lang B).
        cell_ix: index of the target cell that will be updated.
        neighbors: Array containing the languages spoken by the target cell neighbors.
        status_a: Status or prestige of the language A.
        vol: Volatility of the system. Determines the location of the
            fixed points. Defaults to 1.0

    Returns:
        The function returns the updated version of the np.array grid.
    """
    (i, j) = cell_ix
    language = grid[i, j]
    # Proportion of neighbours speaking language A and language B
    n_a = (neighbors > 0).sum() / 4.0
    n_b = 1.0 - n_a
    change_prob = ((1.0 - status_a) * n_b ** vol) if language == 1 else (status_a * n_a ** vol)
    grid[i, j] = -1 * language if np.random.uniform() < change_prob else language
    return grid
