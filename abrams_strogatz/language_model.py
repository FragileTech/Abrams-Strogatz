"""Abrams-strogatz language competition model."""
import numpy as np


def initialize_grid(height: int, width: int, prob_a: float = 0.5) -> np.ndarray:
    """
    Create the initial (height x width) array.

    Each point within the (height x width) array represents a citizen speaking
    language A or language B. The default initial condition corresponds
    to a scenario where the speakers are randomly distributed, the
    probability of each speaker being p(A) = pa and p(B) = 1 - prob_a.

    Args:
        height: Size of the array.
        width: Size of the array.
        prob_a: Probability that a single node within the array speaks
            language A. Defaults to 0.5.

    Returns:
        Returns a np.array(shape=(height, width)) where each node speaks either
        language A (represented by a value 1) or language B (represented
        by a value -1)

    """
    mask = np.random.rand(height, width)
    grid = np.where(mask < prob_a, 1, -1)
    return grid


def periodic_boundary(index: tuple, lattice_shape: tuple) -> tuple:
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

    >>> print(periodic_boundary((6, 5), (5, 5)))
    (1,0)
    """
    return tuple((i % s for i, s in zip(index, lattice_shape)))


def language_dynamics(population: np.array, status_a: float = 0.5, vol: float = 1.0) -> np.array:
    """
    Update one node of the grid containing a population of speakers following the\
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
        population: Array containing the language spoken by the population.
            The values contained inside the array are 1 (lang A) and
            -1 (lang B).
        status_a: Status or prestige of the language A.
        vol: Volatility of the system. Determines the location of the
            fixed points. Defaults to 1.0

    Returns:
        The function returns the updated version of the np.array population.
    """
    width, height = population.shape
    i, j = np.random.randint(width), np.random.randint(height)  # Select random speaker from grid
    language = population[i, j]
    neighbors_ix = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))  # Indexes of speaker neighbors
    neighbor_langs = [population[periodic_boundary(ix, (width, height))] for ix in neighbors_ix]
    # Proportion of neighbours speaking language A and language B
    n_a = (np.array(neighbor_langs) > 0).sum() / 4.0
    n_b = 1.0 - n_a
    change_prob = ((1.0 - status_a) * n_b ** vol) if language == 1 else (status_a * n_a ** vol)
    population[i, j] = -1 * language if np.random.uniform() < change_prob else language
    return population


def plot_grid(population: np.array, close: bool = False):
    """
    Plot a population grid assigning.

    Graphical 2D-representation of the (mxn) array. Each site represents
    an individual speaking either language A or language B. Languages
    are pictured by a binary selection of colors (blue, red).

    Args:
        population: Array containing the language spoken by the population.
            The values contained inside the array are 1 (lang A) and
            -1 (lang B).
        close: if True do not display the figure and only return the figure object.

    Returns:
        Returns a mpl.figure object containing the graphical representation
            of the array.

    """
    import matplotlib
    import matplotlib.pyplot as plt

    cmap = matplotlib.colors.ListedColormap(["Blue", "Red"])
    colbar_tick = np.array([-1, 1])
    fig = plt.figure()
    ax = plt.axes()
    plot = ax.matshow(population, cmap=cmap, origin="lower", animated=True)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    ax.yaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    fig.colorbar(plot, ax=ax, ticks=colbar_tick, label="Language").ax.set_yticklabels(["B", "A"])
    if close:
        # To avoid an excessive computation cost, the graphical
        # representation of the lattice is not displayed. Only the figure instance
        # is returned.
        plt.close()
    return fig
