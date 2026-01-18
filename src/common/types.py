from typing import TypeAlias
import numpy as np
import numpy.typing as npt

Matrix: TypeAlias = npt.NDArray[np.float64]
Vector: TypeAlias = npt.NDArray[np.float64]
Model: TypeAlias = dict[str, Matrix]
