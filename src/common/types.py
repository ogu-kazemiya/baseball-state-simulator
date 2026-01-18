from typing import TypeAlias
import numpy as np
import numpy.typing as npt

Model: TypeAlias = dict[str, npt.NDArray[np.float64]]
Matrix: TypeAlias = npt.NDArray[np.float64]
Vector: TypeAlias = npt.NDArray[np.float64]
