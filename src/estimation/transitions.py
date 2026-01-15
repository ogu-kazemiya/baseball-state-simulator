import numpy as np
import numpy.typing as npt
import pandas as pd
import src.common.constants as consts

def compute_count_matrix(df: pd.DataFrame, events: list[str] | None = None) -> npt.NDArray[np.int64]:
    if events is None:
        events = consts.ALL_EVENTS
    if not set(events).issubset(set(consts.ALL_EVENTS)):
        raise ValueError("Events list contains invalid event types.")
    
    df_filtered = df[df["events"].isin(events)]
    counts = pd.crosstab(df_filtered["state"], df_filtered["next_state"])
    counts = counts.reindex(index=range(25), columns=range(25), fill_value=0)
    count_matrix = counts.to_numpy(dtype=np.int64)
    return count_matrix

def compute_transition_matrix(count_matrix: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    count_matrix = count_matrix.astype(np.float64)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(
        count_matrix,
        row_sums,
        out=np.zeros_like(count_matrix),
        where=row_sums != 0
    )
    prob_matrix[24, :] = 0.0
    prob_matrix[24, 24] = 1.0
    return prob_matrix
