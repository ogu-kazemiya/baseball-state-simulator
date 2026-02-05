from typing import Literal
import numpy as np
import pandas as pd
from .types import Matrix
from .constants import STATE_STR_MAP

def normalize_transition_matrix(matrix: Matrix) -> Matrix:
    mat = matrix.copy()

    row_sums = mat.sum(axis=1, keepdims=True)
    invalid_rows = np.where(row_sums <= 0)[0]

    if invalid_rows.size > 0:
        raise ValueError(
            f"Invalid transition matrix: Rows {invalid_rows} have sum <= 0. "
            "Cannot normalize."
        )

    return mat / row_sums

def print_matrix_formatted(
    matrix: Matrix,
    title: str = "Transition Matrix",
    mode: Literal["count", "rate"] = "rate",
) -> None:
    if matrix.shape != (25, 25):
        raise ValueError("Matrix must be of shape (25, 25) to print formatted.")

    labels = [STATE_STR_MAP[i] for i in range(25)]
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    print(f"=== {title} ===")
    with pd.option_context('display.max_rows', 30, 'display.max_columns', 30, 'display.width', 1000):
        if mode == "count":
            display_df = df.astype(int).replace(0, "")
        elif mode == "rate":
            display_df = df.map(_format_rate)
        else:
            raise ValueError(f"Invalid mode: {mode} (must be 'count' or 'rate')")
        print(display_df)

def _format_rate(val: float) -> str:
    if val == 0 or pd.isna(val):
        return ""

    s = f"{val:.3f}"
    if s.startswith("0"):
        s = s[1:]
    return s
