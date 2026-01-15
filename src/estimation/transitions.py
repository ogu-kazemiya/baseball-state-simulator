import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import src.common.constants as consts

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

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

def save_event_counts(df: pd.DataFrame) -> None:
    count_matrices = {}
    for event in consts.ALL_EVENTS:
        matrix = compute_count_matrix(df, events=[event])
        count_matrices[event] = matrix

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = ARTIFACTS_DIR / "event_count_matrices.npz"
    np.savez_compressed(out_path, **count_matrices)
