import numpy as np
import numpy.typing as npt

def normalize_transition_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mat = matrix.copy()

    row_sums = mat.sum(axis=1, keepdims=True)
    invalid_rows = np.where(row_sums <= 0)[0]

    if invalid_rows.size > 0:
        raise ValueError(
            f"Invalid transition matrix: Rows {invalid_rows} have sum <= 0. "
            "Cannot normalize."
        )

    return mat / row_sums
