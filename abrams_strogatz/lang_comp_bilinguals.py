"""This module implements the Minett-Wang bilingual model."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Initial condition. Randomly distributed speakers.


def initial_cond(m: int, n: int, pa: float = 1./3., pb: float = 1./3.) -> np.array:
    """Create the initial (mxn) array representing a population of speakers.

    Each point within the (mxn) array represents a citizen speaking
    language A, language B, or language A and B. The default initial 
    condition corresponds to a scenario where the speakers are 
    randomly distributed, the probability of each speaker being 
    p(A) = pa, p(B) = pb, and p(AB) = 1. - pa - pb.

    Args:
        m: Size of the array.
        n: Size of the array.
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
    popu = np.random.choice([1, 0, -1], size=(m, n), p=[pa, 1.0 - pa - pb, pb])
    return popu


def repres(popu: np.array) -> plt.figure:
    """Graphical 2D-representation of the (mxn) array. 

    Each site represents an individual speaking either language A, 
    language B, or language A and B (bilinguals). Languages are pictured
    by a selection of colors (blue, white, red).

    Args:
        popu: Array containing the language spoken by the population.
            The values contained inside the array are 1 (lang A), 
            0 (bilingual) and -1 (lang B).

    Returns:
        Returns a mpl.figure object containing the graphical representation
            of the array.

    """
    col = mpl.colors.ListedColormap(["Blue", "White", "Red"])
    colbar_tick = np.array([-1, 0, 1])
    fig = plt.figure()
    ax = plt.axes()
    plot = ax.matshow(popu, cmap=col, origin="lower", animated=True)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    ax.yaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    fig.colorbar(plot, ax=ax, ticks=colbar_tick, label="Language").ax.set_yticklabels(
        ["B", "AB", "A"],)
    # To avoid an excessive computation cost, the graphical
    # representation of the lattice is not displayed. Only the figure instance
    # is returned.
    plt.close()
    return fig


def periodic_boundary(index: tuple, lattice_shape: tuple) -> tuple:
    """Apply periodic boundary conditions.

    We consider a regular lattice with periodic boundary conditions.
    periodic_boundary function is used to apply this condition to 
    the selected node. If the node is located outside the boundaries, 
    the neighbors are selected applying usual nearest neighbor 
    conditions. If the node is located at the boundary, the function 
    will return a tuple where the neighbors are selected following 
    periodic boundary conditions (in this way, the regular lattice 
    becomes a torus).

    Args:
        index: 2D-tuple. Represents the selected node.
        lattice_shape: 2D-tupple. Shape of the lattice.
    Returns:
        The functions returns a tuple where periodic boundary conditions
        are applied (if needed).

    Examples:
    If the lattice is a 5x5 array, a node located at position (6, 5) will
    result:

    >>> print(periodic_boundary((6, 5), (5, 5)))
    (1,0)
    """
    return tuple((i % s for i, s in zip(index, lattice_shape)))


def language_dynamics(popu: np.array, m: int, n: int, s: float, a: float = 1.0) -> np.array:
    """Population evolution. Change the language spoken by individuals. 

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
        popu: Array containing the language spoken by the population.
            The values contained inside the array are 1 (lang A), 
            0 (bilingual) and -1 (lang B).
        m: Size of the array.
        n: Size of the array.
        s: Status or prestige of the language A.
        a: Volatility of the system. Determines the location of the
            fixed points. Defaults to 1.0

    Returns:
        The function returns the updated version of the np.array popu.
    """
    # Selection random node.
    ii, jj = np.random.randint(m), np.random.randint(n)
    lang = popu[ii, jj]
    # Neighbors language
    nn = np.array([])  # Array containing the neighbors
    nn = np.append(nn, popu[periodic_boundary((ii - 1, jj), (m, n))])  # upper
    nn = np.append(nn, popu[periodic_boundary((ii + 1, jj), (m, n))])  # lower
    nn = np.append(nn, popu[periodic_boundary((ii, jj - 1), (m, n))])  # left
    nn = np.append(nn, popu[periodic_boundary((ii, jj + 1), (m, n))])  # right
    # Number of speakers
    nA = (nn == 1.0).sum() / 4.0   # Number A speakers
    nB = (nn == -1.0).sum() / 4.0  # Number B speakers
    nAB = 1.0 - nA - nB            # Number AB speakers 
    # Language dynamics
    # If lang = 1 => prob(A->AB). If lang = -1 => prob(B->AB). If lang = 0 => prob(AB->A) and
    # prob(AB->B)
    if lang == 1:
        popu[ii, jj] = 0 if (np.random.uniform() < ((1.0 - s) * nB ** a)) else popu[ii, jj]
    elif lang == -1: 
        popu[ii, jj] = 0 if (np.random.uniform() < (s * nA ** a)) else popu[ii, jj]
    else:
        prob_change1 = s * (nA + nAB) ** a          # Change AB -> A 
        prob_change2 = (1 - s) * (nB + nAB) ** a    # Change AB -> B
        u, v = np.random.uniform(), np.random.uniform()
        if (u > prob_change1) and (v > prob_change2):
            popu[ii, jj]    # The condition is not fulfilled. 
        else:
            # Can occur that both p(AB->A) and p(AB->B) are satisfied at the same time. 
            # We have to compute again the probability of change until only one of the conditions
            # is satisfied. 
            while (u < prob_change1) and (v < prob_change2):
                u, v = np.random.uniform(), np.random.uniform()
            popu[ii, jj] = 1 if (u < prob_change1) else -1

    return popu


def saving_process(popu: np.array, ph: int, key: bool = True):
    """
    Generate and save language grid plots.

    Speakers are represented in a grid. During the iterative process,
    the program will check if the selected node must change his/her
    language. After a number of iterations (selected by the developer),
    this function saves a snapshot of the language dynamics.

    Args:
        popu: Array containing the language spoken by the population.
            The values contained inside the array are 1 (lang A), 
            0 (bilingual) and -1 (lang B).
        ph: Integer number labelling the representation. Describes the
            number of finished iterations (iterations = ph * mult).
        key: Boolean value. If the key = True, the program calls the
            function. Defaults to True

    Returns:
        The function saves a png file inside the directory where the
        program is executed.

    """
    if key:
        figr = repres(popu)
        figr.savefig("lang_comp_{}.png".format(str(ph)), bbox_inches="tight")


# Parameters
m = 20
n = 20
a = 1.0
s = 0.5
pa = 1./3
pb = 1./3
# Initial condition
popu = initial_cond(m, n, pa, pb)
initial_popu = popu.copy()
# Check error
na = np.count_nonzero(popu[popu > 0])
nb = np.count_nonzero(popu[popu < 0])
nab = (popu == 0).sum()
print("first error") if nb != (m * n - na - nab) else None
print("num_A = {0}, num_B = {1}, num_AB = {2}".format(na, nb, nab)) 
# Iterative process
mult = 15
maxit = mult * 400  # Maximum number of iterations. Information is stored
# every mult steps
num_speakers = np.ones(shape=(int(maxit / mult), 3))  # Container

for iterr in np.arange(maxit):
    popu = language_dynamics(popu, m, n, s)
    # Saving process and number of speakers
    if (iterr % mult) == 0:
        ph = int(iterr / mult)
        numa = np.count_nonzero(popu[popu > 0])
        numb = np.count_nonzero(popu[popu < 0])
        numab = (popu == 0).sum()
        print("error") if numb != (m * n - numa - numab) else None
        num_speakers[ph] = np.array([numa, numb, numab])
        saving_process(popu, ph, key=False)
    else:
        None

# Representation
color = mpl.colors.ListedColormap(["Blue", "White", "Red"])
colbar_tick = np.array([-1, 0, 1])
ratio = popu.shape[0] / popu.shape[1]
# Figures
fig, axs = plt.subplots(nrows=2, ncols=2)
gs = axs[1, 0].get_gridspec()
fig.suptitle("Language competition")
# First and second plots
for col in range(2):
    ax = axs[0, col]
    grid = initial_popu if (col == 0) else popu
    title = "Initial state" if (col == 0) else "Final state"
    plot = ax.matshow(grid, cmap=color, origin="lower", animated=True)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    ax.yaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
    ax.set_title(title)
    fig.colorbar(
        plot, ax=ax, ticks=colbar_tick, label="Language", fraction=0.047 * ratio,
    ).ax.set_yticklabels(["B", "AB", "A"])
# Third plot
for ax in axs[1, :]:
    ax.remove()
axbig = fig.add_subplot(gs[1, :])
axbig.plot(
    num_speakers[:, 0],
    c="red",
    linestyle="solid",
    marker="o",
    linewidth=0.5,
    markersize=3.5,
    label="Lang A",
)
axbig.plot(
    num_speakers[:, 1],
    c="blue",
    linestyle="solid",
    marker="x",
    linewidth=0.5,
    markersize=3.5,
    label="Lang B",
)
axbig.plot(
    num_speakers[:, 2],
    c="Gray",
    linestyle="solid",
    marker="^",
    linewidth=0.5,
    markersize=3.5,
    label="Lang AB",
)
axbig.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) * mult))
axbig.set(
    xlabel="Iteration",
    ylabel="Number speakers",
    title="Number of speakers as a function of time",
)
axbig.legend(loc="upper left")
plt.tight_layout()
plt.show()
