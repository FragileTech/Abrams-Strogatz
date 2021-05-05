"""Module containing  custom classes to study different language competition models."""
from typing import Optional, Tuple, Union

from bokeh.models import FixedTicker
import holoviews as hv
from holoviews import opts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import panel as pn 

import language_competition.abrams_strogatz as abrs
import language_competition.minett_wang as mw


hv.extension("bokeh")
pn.extension()

class SpeakersGrid:
    """
    Initialize a grid containing individuals speaking different languages.

    Instances of this class contain Numpy arrays in which each node represents
    an individual speaking a particular language (the language spoken
    by the node depends on the value assigned to it).
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        n_languages: int,
        data: Optional[np.ndarray] = None,
    ):
        """
        Initialize the class.

        Initialize an instance with the following attributes:
            width: parameter describing the number of columns of the array.
            height: parameter describing the number of rows of the array.
            shape: tuple formed by the attributes (height, width).
            n_languages: number of distinct languages spoken on the grid.
            data: np.array representing the language spoken by each node.

        Args:
            shape: 2D-tuple containing the height (number of rows) and
                width (number of columns) of the array.
            n_languages: Integer number parametrizing the number of
                distinct languages.
            data: Optional parameter that contains a np.array describing
                the languages spoken by each node.

        Return:
            Returns an object with the aforementioned parameters as
                attributes.
        """
        self._data = data
        height, width = shape
        self._width = width
        self._height = height
        self._n_languages = n_languages

    @property
    def width(self) -> int:
        """Describe the number of columns of the array."""
        return self._width

    @property
    def height(self) -> int:
        """Describe the number of rows of the array."""
        return self._height

    @property
    def n_languages(self) -> int:
        """Parametrize the number of distinct languages."""
        return self._n_languages

    @property
    def shape(self) -> Tuple[int, int]:
        """Contain the (height, width) of the array."""
        return self.height, self.width

    @property
    def data(self) -> Union[np.ndarray, None]:
        """Contain a np.array describing the languages spoken by each node."""
        return self._data

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "SpeakersGrid":
        """Return a SpeakersGrid object created with the attributes of the introduced np.array."""
        return SpeakersGrid(shape=data.shape, n_languages=len(np.unique(data)), data=data)

    def __getitem__(self, item) -> Union[int, np.ndarray]:
        """Allow accessing the values contained inside _data[item]."""
        if self._data is None:
            raise KeyError("Grid no initialized.")
        elif isinstance(item, tuple):
            return self._data[item[0], item[1]]
        return self._data[item]

    def __setitem__(self, key, value) -> None:
        """Allow updating the value contained inside _data[key] with a new constant."""
        if self._data is None:
            raise KeyError("Grid no initialized.")
        elif isinstance(key, tuple) and len(key) == 2:
            self._data[key[0], key[1]] = value
        else:
            self._data[key] = value

    def __str__(self) -> str:
        """Display information of the class instance."""
        return f"{self.__class__.__name__} shape: {self.shape} n_languages: {self.n_languages}"

    def update(self, data: Union["SpeakersGrid", np.ndarray]):
        """Update the information contained in attribute _data."""
        if isinstance(data, SpeakersGrid):
            self._data = data._data.copy()
        else:
            self._data = data.copy()

    def periodic_boundary(self, index: Tuple[int, int]) -> Tuple[int, int]:
        """
        Apply periodic boundary conditions.

        We consider a regular lattice with periodic boundary conditions.
        This function applies this condition to the introduced index.
        If the node is located far from the boundaries, its neighbors
        are selected applying the usual nearest neighbor conditions. If
        the node is located at the boundary, the function will return a
        tuple where the neighbors are selected following periodic
        boundary conditions (in this way, our regular lattice becomes a
        torus).

        Args:
            self: SpeakersGrid instance from which we obtain the shape of
                the lattice.
            index: 2D-tuple. Represents the selected node.
        Returns:
            The functions returns a tuple where periodic boundary conditions
            are applied (if needed).

        Examples:
            If the lattice is a 5x5 array, a node located at position (6, 5) will
            result:

        >>> periodic_boundary((6, 5), (5, 5))
        (1,0)
        """
        return tuple((i % s for i, s in zip(index, self.shape)))

    def sample_random_cell(self) -> Tuple[int, int]:
        """
        Return the index of a grid cell sampled at random.

        Given the shape of the lattice (shape = height, width), it
        returns a 2D-tuple containing a lattice node selected at random.
        """
        height, width = self.shape
        j, i = np.random.randint(height), np.random.randint(width)
        return j, i

    def neighbors(self, ix: Tuple[int, int], indexes: bool = False) -> np.ndarray:
        """
        Return the cell values of the neighbors of the target cell.

        Given an index, it computes which language are spoken by the
        neighbors surrounding the given node.

        Args:
            self: SpeakersGrid instance from which we obtain the values
                of each node in the lattice.
            ix: 2D-tuple containing the selected node to which the
                function will compute the languages spoken by its neighbors.
            indexes: Boolean value. When True, the function returns the
                neighbors indexes along with their values. Defaults to False.

        Return:
            Returns the neighbor values of the selected node. If indexes
                is True, the function also returns their indexes.
        """
        j, i = ix  # height and width
        neighbors = ((j, i - 1), (j, i + 1), (j - 1, i), (j + 1, i))  # left, right, upper, down
        values = np.array([self.data[self.periodic_boundary(neighbor)] for neighbor in neighbors])
        return (neighbors, values) if indexes else values


class LanguageModel:
    """
    Evolve a grid containing nodes speaking different languages.

    Given a particular grid containing individuals (located at each lattice
    site) speaking different languages, this class contains the necessary
    methods to evolve the population according to a given model (Abrams-
    Strogatz or Minett-Wang).
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        n_languages: int,
        data: Optional[np.ndarray] = None,
        grid: Optional[SpeakersGrid] = None,
    ):
        """
        Initialize the class.

        Initialize an instance with the following attributes:
            grid: SpeakersGrid object.
            memory: container where the different iterations of grid are stored.
            width: parameter describing the number of columns of the array.
            height: parameter describing the number of rows of the array.
            shape: tuple formed by the attributes (height, width).
            n_languages: number of distinct languages spoken on the grid.
            data: np.array representing the language spoken by each node.

        Args:
            shape: 2D-tuple containing the height (number of rows) and
                width (number of columns) of the array.
            n_languages: Integer number parametrizing the number of
                distinct languages.
            data: Optional parameter that contains a np.array describing
                the languages spoken by each node.

        Return:
            Returns an object with the aforementioned parameters as
                attributes.
        """
        self.grid = (
            SpeakersGrid(shape=shape, n_languages=n_languages, data=data) if grid is None else grid
        )
        self._memory = []

    @property
    def width(self) -> int:
        """Describe thenumber of columns of the array."""
        return self.grid.width

    @property
    def height(self) -> int:
        """Describe the number of rows of the array."""
        return self.grid.height

    @property
    def n_languages(self) -> int:
        """Parametrize the number of distinct languages."""
        return self.grid.n_languages

    @property
    def shape(self) -> Tuple[int, int]:
        """Contain the (height, width) of the array."""
        return self.height, self.width

    @property
    def memory(self) -> np.ndarray:
        """Store the different iterations of grid."""
        return np.stack(self._memory)

    @property
    def speakers(self) -> Tuple[int, ...]:
        """Count the number of speakers of each language."""
        raise NotImplementedError()

    def __repr__(self) -> None:
        """Display a graphical representation of the lattice."""
        raise NotImplementedError()

    def reset(self) -> SpeakersGrid:
        """Create the initial grid."""
        raise NotImplementedError()

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        """Evolve the grid."""
        raise NotImplementedError()

    def step(
        self,
        grid: Optional[Union[SpeakersGrid, np.ndarray]] = None,
        track: bool = False,
    ) -> SpeakersGrid:
        """
        Update the values contained inside the grid.

        Each time this function is called, it calls the function
        'evolve_grid', which computes the probability that the selected
        node changes its language. This function updates the values
        contained inside the _data attribute according to the chosen
        model (either Abrams-Strogatz or Minett-Wang).

        Args:
            grid: Lattice representing the population and their languages.
            track: Boolean value. If True, the lattice stored in grid is
                saved in _memory.

        Returns:
            Returns a lattice containing the updated values.
        """
        if grid is not None:
            self.grid.update(grid)
        ix = self.grid.sample_random_cell()
        self.evolve_grid(cell_ix=ix)
        if track:
            self._memory.append(self.grid.data)
        return self.grid

    def run(self, epochs: int, track: bool = False) -> SpeakersGrid:
        """
        Start the grid and iterate the model.

        This function initializes the lattice (following the chosen
        model) and evolves it during a specific number iterations.

        Args:
            epochs: Number of iterations.
            track: Boolean value. If True, the lattice stored in grid is
                saved in _memory.

        Returns:
            Gives the resulting lattice, where the values at the lattice
                sites has been updated.
        """
        self.reset()  # Initialize the model
        for _ in range(epochs):
            self.step(track=track)
        if track:
            self._plot()
        return self.grid

    def _plot(self) -> hv.Element:
        """
        Display information on the initial and final state of the lattice.

        Graphical representation of the lattice. The image shows the
        initial and final state of the grid (in order to compare how
        the network has evolved), as well as the number of speakers
        as a function of time.

        In order to apply this method, the values contained inside the
        lattice must have been stored inside self._memory.
        """
        raise NotImplementedError()


class AbramsStrogatz(LanguageModel):
    """
    Evolve the lattice following the Abrams-Strogatz model.

    This class applies the Abrams-Strogatz model to study the evolution
    of  two languages. Given a lattice site, it computes the probability
    that the former changes its language. The number of languages is two;
    language A (represented by 1) and language B (represented by -1).
    """

    N_LANGUAGES = 2

    def __init__(
        self,
        shape: Tuple[int, int],
        data: Optional[np.ndarray] = None,
        status_a: float = 0.5,
        vol: float = 1.0,
        prob_a0: float = 0.5,
    ):
        """
        Initialize the Abrams-Strogatz language competition model.

        Args:
            shape: (height, width) of the model grid.
            status_a: Status or prestige of the language A.
            vol: Volatility of the system. Determines the location of the
                fixed points. Defaults to 1.0
            prob_a0: Probability that a single node within the array speaks
                language A. Defaults to 0.5.
        """
        super(AbramsStrogatz, self).__init__(shape=shape, n_languages=self.N_LANGUAGES, data=data)
        self.status_a = status_a
        self.vol = vol
        self.prob_a0 = prob_a0

    @property
    def speakers(self) -> Tuple[int, int]:
        """Count the number of speakers of each language."""
        num_a = (self.grid.data > 0).sum()
        num_b = (self.grid.data < 0).sum()
        return (num_a, num_b)

    def __repr__(self) -> None:
        """
        Display a graphical representation of the lattice.

        Graphical 2D-representation of the (mxn) array. Each site represents
        an individual speaking either language A or language B. Languages
        are pictured by a binary selection of colors (blue, red).
        """
        col = mpl.colors.ListedColormap(["Blue", "Red"])
        colbar_tick = np.array([-1, 1])
        fig = plt.figure()
        ax = plt.axes()
        plot = ax.matshow(self.grid.data, cmap=col, origin="lower", animated=True)
        ax.set_title("Abrams-Strogatz model")
        ax.xaxis.tick_bottom()
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
        ax.yaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
        fig.colorbar(plot, ax=ax, ticks=colbar_tick, label="Language").ax.set_yticklabels(
            ["B", "A"],
        )
        plt.show()
        return ""

    def reset(self) -> SpeakersGrid:
        """
        Create the initial (height x width) array.

        Each point within the (height x width) array represents a citizen
        speaking language A or language B. The default initial condition
        corresponds to a scenario where the speakers are randomly distributed,
        the probability of each speaker being p(A) = prob_a0 and
        p(B) = 1 - prob_a0.

        self.grid will be a np.array(shape = (height, width)), where each
        node speaks either language A (represented by a value 1) or
        language B (represented by a value -1)
        """
        data = abrs.initialize_grid(self.shape, prob_a=self.prob_a0)
        self.grid = self.grid.from_numpy(data)
        self._memory = [data.copy()]
        return self.grid

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        """
        Update one lattice site following the Abrams-Strogatz language model.

        Each time this function is called, it computes the probability of
        change for a selected node.
        The probability to change from language A to B is:
        pAB = (1 - status_a) * nB ** vol
        The probability to change from language B to A is:
        pBA = status_a * nA ** vol

        Args:
            cell_ix: index of the target cell that will be updated.

        Returns:
            The function returns the updated version of the np.array grid.
        """
        new_data = abrs.update_grid(
            grid=self.grid.data,
            cell_ix=cell_ix,
            neighbors=self.grid.neighbors(cell_ix),
            status_a=self.status_a,
            vol=self.vol,
        )
        self.grid.update(new_data)

    def _plot(self) -> hv.Element:
        """
        Represent the initial and final state of the lattice.

        Graphical representation of the lattice. The image shows the
        initial and final state of the grid (in order to compare how
        the network has evolved), as well as the number of speakers
        as a function of time.
        """
        grid_flat = self.memory.reshape(self.memory.shape[0], -1)
        speakers_a = (grid_flat > 0).sum(1)
        speakers_b = (grid_flat < 0).sum(1)
        total = speakers_a + speakers_b == (self.width * self.height) * np.ones(len(self.memory))
        if not np.all(total):
            raise ValueError(
                "The total number of speakers does not correspond to the lattice size!",
            )
        # Plots
        colors = ["red", "blue"]
        data_start = self.memory[0]
        data_end = self.grid.data
        grid_start = {
            "xdata": np.arange(1, data_start.shape[0] + 1),
            "ydata": np.arange(1, data_start.shape[1] + 1),
            "zdata": data_start,
        }
        grid_end = {
            "xdata": np.arange(1, data_end.shape[0] + 1),
            "ydata": np.arange(1, data_end.shape[1] + 1),
            "zdata": data_end,
        }
        plot_start = hv.Image(
            grid_start,
            kdims=["xdata", "ydata"],
            vdims=hv.Dimension("zdata", range=(-1, 1)),
            group="grid", 
            label="Initial"
        )
        plot_end = hv.Image(
            grid_end,
            kdims=["xdata", "ydata"],
            vdims=hv.Dimension("zdata", range=(-1, 1)),
            group="grid", 
            label="Final"
        )
        plot_curvea = hv.Curve(
            speakers_a,
            label="Speakers A",
        ).opts(color="red")
        plot_curveb = hv.Curve(
            speakers_b,
            label="Speakers B",
        ).opts(color="blue")
        # Compositions
        lines = plot_curvea * plot_curveb
        layout = pn.Column(pn.Row(plot_start, plot_end), lines)
        # Options
        layout[1].opts(
            opts.Image(
                invert_yaxis=True,
                cmap=colors,
                colorbar=True,
                width=350,
                labelled=[],
                colorbar_opts={
                    "title": "Languages",
                    "title_text_align": "left",
                    "major_label_overrides": {-0.5: "B", 0.5: "A"},
                    "ticker": FixedTicker(ticks=[-0.5, 0.5]),
                    "major_label_text_align": "right",
                },
            ),
            opts.Curve(xlabel="Iterations", ylabel="Number of speakers", width=700),
        )
        layout.cols(2)
        return layout.cols(2)


class MinettWang(AbramsStrogatz):
    """
    Evolve the lattice following the Minett-Wang model.

    It applies the rules of the Minett-Wang model. Given a lattice
    site, it computes the probability that the former changes its language.
    The number of languages is three; language A (represented by 1),
    language B (represented by -1) and language AB or bilinguals (represented
    by 0).
    """

    N_LANGUAGES = 3

    def __init__(
        self,
        shape: Tuple[int, int],
        status_a: float = 0.5,
        vol: float = 1.0,
        prob_a0: float = 0.33,
        prob_b0: float = 0.33,
    ):
        """
        Initialize a Minett-Wang language competition model.

        Args:
            shape: (height, width) of the model grid.
            status_a: Status or prestige of the language A.
            vol: Volatility of the system. Determines the location of the
             fixed points. Defaults to 1.0
            prob_a0: Probability that a single node within the array speaks
                language A. Defaults to 0.33.
            prob_b0: Probability that a single node within the array speaks
                language B. Defaults to 0.33.
        """
        self.prob_b0 = prob_b0
        super(MinettWang, self).__init__(shape=shape, status_a=status_a, vol=vol, prob_a0=prob_a0)

    @property
    def speakers(self) -> Tuple[int, int, int]:
        """Count the number of speakers of each language."""
        num_a = (self.grid.data > 0).sum()
        num_b = (self.grid.data < 0).sum()
        num_ab = (self.grid.data == 0).sum()
        return (num_a, num_b, num_ab)

    def __repr__(self) -> None:
        """
        Display a graphical representation of the lattice.

        Graphical 2D-representation of the (mxn) array. Each site represents
        an individual speaking either language A, language B, or language A
        and B (bilinguals). Languages are pictured by a selection of colors
        (blue, white, red).
        """
        col = mpl.colors.ListedColormap(["Blue", "White", "Red"])
        colbar_tick = np.array([-1, 0, 1])
        fig = plt.figure()
        ax = plt.axes()
        plot = ax.matshow(self.grid.data, cmap=col, origin="lower", animated=True)
        ax.set_title("Minett-Wang model")
        ax.xaxis.tick_bottom()
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.xaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
        ax.yaxis.set_major_formatter(lambda val, pos: r"{}".format(int(val) + 1))
        fig.colorbar(plot, ax=ax, ticks=colbar_tick, label="Language").ax.set_yticklabels(
            ["B", "AB", "A"],
        )
        plt.show()
        return ""

    def reset(self) -> SpeakersGrid:
        """
        Create the initial (height x width) array.

        Each point within the (height x width) array represents a citizen
        speaking language A, language B or language AB (bilinguals). The
        default initial condition corresponds to a scenario where the speakers
        are randomly distributed, the probability of each speaker being
        p(A) = prob_a0, p(B) = prob_b0 and p(AB) = 1 - prob_a0 - prob_b0.

        self.grid will be a np.array(shape = (height, width)), where each
        node speaks either language A (represented by a value 1), language B
        (represented by a value -1) or language AB (represented by a value 0).
        """
        data = mw.initialize_grid(shape=self.shape, pa=self.prob_a0, pb=self.prob_b0)
        self.grid = self.grid.from_numpy(data)
        self._memory = [data.copy()]
        return self.grid

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        """
        Update one lattice site following the Minett-Wang language model.

        Each time this function is called, it computes the probability of
        change for a selected node.
        The probability to change from language A to AB is:
        p(A->AB) = (1 - status_a) * nB**vol
        The probability to change from language B to AB is:
        p(B->AB) = status_a * nA**vol
        The probability to change from language AB to A is:
        p(AB->A) = status_a * (nA + nAB)**vol
        The probability to change from language AB to B is:
        p(B->AB) = (1 - status_a) * (nB + nAB)**vol

        Args:
            cell_ix: index of the target cell that will be updated.

        Returns:
            The function returns the updated version of the np.array grid.
        """
        new_data = mw.update_grid(
            grid=self.grid.data,
            cell_ix=cell_ix,
            neighbors=self.grid.neighbors(cell_ix),
            status_a=self.status_a,
            vol=self.vol,
        )
        self.grid.update(new_data)

    def _plot(self) -> hv.Element:
        """
        Represent the initial and final state of the lattice.

        Graphical representation of the lattice. The image shows the
        initial and final state of the grid (in order to compare how
        the network has evolved), as well as the number of speakers
        as a function of time.
        """
        grid_flat = self.memory.reshape(self.memory.shape[0], -1)
        speakers_a = (grid_flat == 1).sum(1)
        speakers_b = (grid_flat == -1).sum(1)
        speakers_ab = (grid_flat == 0).sum(1)
        total = speakers_a + speakers_b + speakers_ab == (self.width * self.height) * np.ones(
            len(self.memory),
        )
        if not np.all(total):
            raise ValueError(
                "The total number of speakers does not correspond to the lattice size!",
            )
        # Plots
        colors = ["red", "white", "blue"]
        data_start = self.memory[0]
        data_end = self.grid.data
        grid_start = {
            "xdata": np.arange(1, data_start.shape[0] + 1),
            "ydata": np.arange(1, data_start.shape[1] + 1),
            "zdata": data_start,
        }
        grid_end = {
            "xdata": np.arange(1, data_end.shape[0] + 1),
            "ydata": np.arange(1, data_end.shape[1] + 1),
            "zdata": data_end,
        }
        plot_start = hv.Image(
            grid_start,
            kdims=["xdata", "ydata"],
            vdims=hv.Dimension("zdata", range=(-1, 1)),
            title="Initial grid",
        )
        plot_end = hv.Image(
            grid_end,
            kdims=["xdata", "ydata"],
            vdims=hv.Dimension("zdata", range=(-1, 1)),
            title="Final grid",
        )
        plot_curvea = hv.Curve(speakers_a, label="Speakers A", color="red")
        plot_curveb = hv.Curve(speakers_b, label="Speakers B", color="blue")
        plot_curveab = hv.Curve(speakers_ab, label="Speakers AB", color="gray")
        # Compositions
        grids = plot_start + plot_end
        lines = plot_curvea * plot_curveb * plot_curveab
        layout = grids + lines
        # Options
        layout.opts(
            opts.Image(
                invert_yaxis=True,
                cmap=colors,
                colorbar=True,
                width=350,
                labelled=[],
                colorbar_opts={
                    "title": "Languages",
                    "title_text_align": "left",
                    "major_label_overrides": {-1: "B", 0: "AB", 1: "A"},
                    "ticker": FixedTicker(ticks=[-1, 0, 1]),
                    "major_label_text_align": "right",
                },
            ),
            opts.Curve(xlabel="Iterations", ylabel="Number of speakers", width=700),
        )
        return layout.cols(1)
