"""This module implements the Minett-Wang bilingual model."""
from typing import Tuple

import numpy as np


# Initial condition. Randomly distributed speakers.


def initialize_grid(
    shape: Tuple[int, int],
    pa: float = 1.0 / 3.0,
    pb: float = 1.0 / 3.0,
) -> np.array:
    """Create the initial (mxn) array representing a grid of speakers.

    Each point within the (mxn) array represents a citizen speaking
    language A, language B, or language A and B. The default initial
    condition corresponds to a scenario where the speakers are
    randomly distributed, the probability of each speaker being
    p(A) = pa, p(B) = pb, and p(AB) = 1. - pa - pb.

    Args:
        shape: (height, width) of the initialized grid.
        pa: Probability that a single node within the array speaks
            language A. Defaults to 0.33.
        pb: Probability that a single node within the array speaks
            language B. Defaults to 0.33.

    Returns:
        Returns a np.array(shape=(m, n)) where each node speaks either
        language A (represented by a value 1), language B (represented
        by a value -1) or laguage A and B (represented by a value 0).
        The latter represent bilingual speakers.

    """
    popu = np.random.choice([1, 0, -1], size=shape, p=[pa, 1.0 - pa - pb, pb])
    return popu


def update_grid(
    grid: np.array,
    cell_ix: Tuple[int, int],
    neighbors: np.ndarray,
    status_a: float,
    vol: float = 1.0,
) -> np.array:
    """Evolve the grid of speakers following the Minett-Wang bilingual model.

    Language dynamics. Evolution of the number of speakers of each
    language. Each time this function is called, it computes the
    probability of change for a selected node.
    The probability to change from language A to AB is:
    p(A->AB) = (1 - s) * nB**a
    The probability to change from language B to AB is:
    p(B->AB) = s * nA**a
    The probability to change from language AB to A is:
    p(AB->A) = s * (nA + nAB)**a
    The probability to change from language AB to B is:
    p(B->AB) = (1 - s) * (nB + nAB)**a
    The steps are the following:
    1) A random node is selected.
    2) The language of each neighbor surrounding the node is computed,
    counting the number of speakers.
    3) The probability of change is computed. If the probability is
    bigger than a uniformly distributed random number, the language
    of the selected node is changed.

    Args:
        grid: Array containing the language spoken by the grid.
            The values contained inside the array are 1 (lang A),
            0 (bilingual) and -1 (lang B).
        cell_ix: index of the target cell that will be updated.
        neighbors: Array containing the languages spoken by the target cell neighbors.
        status_a: Status or prestige of the language A.
        vol: Volatility of the system. Determines the location of the
            fixed points. Defaults to 1.0

    Returns:
        The function returns the updated version of the np.array popu.
    """
    (i, j) = cell_ix
    lang = grid[i, j]
    # Number of speakers
    nA = (neighbors == 1.0).sum() / 4.0  # Number A speakers
    nB = (neighbors == -1.0).sum() / 4.0  # Number B speakers
    nAB = 1.0 - nA - nB  # Number AB speakers
    # Language dynamics
    # If lang = 1 => prob(A->AB). If lang = -1 => prob(B->AB). If lang = 0 => prob(AB->A) and
    # prob(AB->B)
    if lang == 1:
        grid[i, j] = 0 if (np.random.uniform() < ((1.0 - status_a) * nB ** vol)) else grid[i, j]
    elif lang == -1:
        grid[i, j] = 0 if (np.random.uniform() < (status_a * nA ** vol)) else grid[i, j]
    else:
        prob_change1 = status_a * (nA + nAB) ** vol  # Change AB -> A
        prob_change2 = (1 - status_a) * (nB + nAB) ** vol  # Change AB -> B
        u, v = np.random.uniform(), np.random.uniform()
        if not ((u > prob_change1) and (v > prob_change2)):
            # Can occur that both p(AB->A) and p(AB->B) are satisfied at the same time.
            # We have to compute again the probability of change until only one of the conditions
            # is satisfied.
            while (u < prob_change1) and (v < prob_change2):
                u, v = np.random.uniform(), np.random.uniform()
            grid[i, j] = 1 if (u < prob_change1) else -1

    return grid
