from typing import Optional, Tuple, Union

import numpy as np

import language_competition.abrams_strogatz as abrs
from language_competition.grid_utils import periodic_boundary, sample_random_cell
import language_competition.minett_wang as mw


class SpeakersGrid:
    def __init__(
        self,
        shape: Tuple[int, int],
        n_languages: int,
        data: Optional[np.ndarray] = None,
    ):
        self._data = data
        height, width = shape
        self._width = width
        self._height = height
        self._n_languages = n_languages

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def n_languages(self) -> int:
        return self._n_languages

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def data(self) -> Union[np.ndarray, None]:
        return self._data

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "SpeakersGrid":
        return SpeakersGrid(shape=data.shape, n_languages=len(np.unique(data)), data=data)

    def update(self, data: Union["SpeakersGrid", np.ndarray]):
        if isinstance(data, SpeakersGrid):
            self._data = data._data.copy()
        else:
            self._data = data.copy()

    def __getitem__(self, item) -> Union[int, np.ndarray]:
        if self._data is None:
            raise KeyError("Grid no initialized.")
        elif isinstance(item, tuple):
            return self._data[item[0], item[1]]
        return self._data[item]

    def __setitem__(self, key, value) -> None:
        if self._data is None:
            raise KeyError("Grid no initialized.")
        elif isinstance(key, tuple) and len(key) == 2:
            self._data[key[0], key[1]] = value
        else:
            self._data[key] = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__} shape: {self.shape} n_languages: {self.n_languages}"

    def sample_random_cell(self) -> Tuple[int, int]:
        return sample_random_cell(self.data)

    def neighbors(self, ix: Tuple[int, int], indexes: bool = False) -> np.ndarray:
        """Return the cell values (spoken languages) of the neighbors of the target cell."""
        i, j = ix
        neighbor_idxs = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        values = np.array([self.data[periodic_boundary(ix, self.shape)] for ix in neighbor_idxs])
        return (neighbor_idxs, values) if indexes else values


class LanguageModel:
    def __init__(
        self,
        shape: Tuple[int, int],
        n_languages: int,
        data: Optional[np.ndarray] = None,
    ):
        self.grid = SpeakersGrid(shape=shape, n_languages=n_languages, data=data)
        self._memory = []

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def n_languages(self) -> int:
        return self.grid.n_languages

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def memory(self) -> np.ndarray:
        return np.stack(self._memory)

    def reset(self) -> SpeakersGrid:
        raise NotImplementedError()

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        raise NotImplementedError()

    def step(
        self,
        grid: Optional[Union[SpeakersGrid, np.ndarray]] = None,
        track: bool = False,
    ) -> SpeakersGrid:
        if grid is not None:
            self.grid.update(grid)
        ix = self.grid.sample_random_cell()
        self.evolve_grid(cell_ix=ix)
        if track:
            self._memory.append(self.grid.data)
        return self.grid

    def run(self, epochs: int, track: bool = False) -> SpeakersGrid:
        self.reset()
        for _ in range(epochs):
            self.step(track=track)
        return self.grid


class AbramsStrogatz(LanguageModel):
    N_LANGUAGES = 2

    def __init__(
        self,
        shape: Tuple[int, int],
        status_a: float = 0.5,
        vol: float = 1.0,
        prob_a0: float = 0.5,
    ):
        """
        Initialize an AbramsStrogatz language competition model.

        Args:
            shape: (height, width) of the model grid.
            status_a:
            vol:
            prob_a0: Probability that a single node within the array speaks
                language A. Defaults to 0.5.
        """
        super(AbramsStrogatz, self).__init__(shape=shape, n_languages=self.N_LANGUAGES)
        self.status_a = status_a
        self.vol = vol
        self.prob_a0 = prob_a0

    def reset(self) -> SpeakersGrid:
        """
        Create the initial (height x width) array.

        Each point within the (height x width) array represents a citizen speaking
        language A or language B. The default initial condition corresponds
        to a scenario where the speakers are randomly distributed, the
        probability of each speaker being p(A) = pa and p(B) = 1 - prob_a.

        self.grid will be anp.array(shape=(height, width)) where each node speaks either \
        language A (represented by a value 1) or language B (represented \
        by a value -1)
        """
        data = abrs.initialize_grid(self.shape, prob_a=self.prob_a0)
        self.grid = self.grid.from_numpy(data)
        self._memory = [data.copy()]
        return self.grid

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        new_data = abrs.update_grid(
            grid=self.grid.data,
            cell_ix=cell_ix,
            neighbors=self.grid.neighbors(cell_ix),
            status_a=self.status_a,
            vol=self.vol,
        )
        self.grid.update(new_data)


class MinettWang(AbramsStrogatz):
    N_LANGUAGES = 3

    def __init__(
        self,
        shape: Tuple[int, int],
        status_a: float = 0.5,
        vol: float = 1.0,
        prob_a0: float = 0.33,
        prob_b0: float = 0.33,
    ):
        self.prob_b0 = prob_b0
        super(MinettWang, self).__init__(shape=shape, status_a=status_a, vol=vol, prob_a0=prob_a0)

    def reset(self) -> SpeakersGrid:
        data = mw.initialize_grid(shape=self.shape, pa=self.prob_a0, pb=self.prob_b0)
        self.grid = self.grid.from_numpy(data)
        self._memory = [data.copy()]
        return self.grid

    def evolve_grid(self, cell_ix: Tuple[int, int]) -> None:
        new_data = mw.update_grid(
            grid=self.grid.data,
            cell_ix=cell_ix,
            neighbors=self.grid.neighbors(cell_ix),
            status_a=self.status_a,
            vol=self.vol,
        )
        self.grid.update(new_data)
