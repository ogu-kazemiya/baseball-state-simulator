import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import src.common.types as types
import src.common.constants as consts
import src.common.matrix_utils as matrix_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

def compute_count_matrices(df: pd.DataFrame) -> dict[str, npt.NDArray[np.int64]]:
    count_matrices: dict[str, npt.NDArray[np.int64]] = {}
    for event in consts.ALL_EVENTS:
        df_filtered = df[df["events"] == event]
        counts = pd.crosstab(df_filtered["state"], df_filtered["next_state"])
        counts = counts.reindex(index=range(25), columns=range(25), fill_value=0)
        count_matrix = counts.to_numpy(dtype=np.int64)
        count_matrices[event] = count_matrix
    return count_matrices

def save_count_matrices(count_matrices: dict[str, npt.NDArray[np.int64]]) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = ARTIFACTS_DIR / "event_count_matrices.npz"
    np.savez_compressed(out_path, **count_matrices)

def create_model(count_matrices: dict[str, npt.NDArray[np.int64]]) -> types.Model:
    model: types.Model = {}

    for result, events in consts.RESULT_MAPPING.items():
        count_matrix = np.zeros((25, 25), dtype=np.int64)
        for event in events:
            count_matrix += count_matrices[event]

        # 合計が0の行を対角成分1に置換
        row_sums = count_matrix.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if len(zero_rows) > 0:
            count_matrix[zero_rows, zero_rows] = 1

        prob_matrix = matrix_utils.normalize_transition_matrix(count_matrix.astype(np.float64))
        model[result] = prob_matrix

    return model

def save_model(model: types.Model, model_name: str) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{model_name}_model.npz"
    np.savez_compressed(out_path, **model)
