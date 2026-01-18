from typing import TypeAlias
import numpy as np
import numpy.typing as npt

Model: TypeAlias = dict[str, npt.NDArray[np.float64]]
TransitionMatrix: TypeAlias = npt.NDArray[np.float64]
StateVector: TypeAlias = npt.NDArray[np.float64]
