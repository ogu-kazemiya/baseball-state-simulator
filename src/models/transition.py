import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from src.common.types import Model
from src.common.constants import PA_EVENTS
from src.common.model_rules import RESULT_MAPPING
from src.common.matrix_utils import normalize_transition_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

def aggregate_count_matrices(df: pd.DataFrame) -> dict[str, npt.NDArray[np.int64]]:
    count_matrices: dict[str, npt.NDArray[np.int64]] = {}
    for event in PA_EVENTS:
        df_filtered = df[df["events"] == event]
        counts = pd.crosstab(df_filtered["state"], df_filtered["next_state"])
        counts = counts.reindex(index=range(25), columns=range(25), fill_value=0)
        count_matrix = counts.to_numpy(dtype=np.int64)
        count_matrices[event] = count_matrix
    return count_matrices

def build_model(count_matrices: dict[str, npt.NDArray[np.int64]]) -> Model:
    model: Model = {}

    for result, events in RESULT_MAPPING.items():
        count_matrix = np.zeros((25, 25), dtype=np.int64)
        for event in events:
            count_matrix += count_matrices[event]

        # 合計が0の行を対角成分1に置換
        row_sums = count_matrix.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if len(zero_rows) > 0:
            count_matrix[zero_rows, zero_rows] = 1

        prob_matrix = normalize_transition_matrix(count_matrix.astype(np.float64))
        model[result] = prob_matrix

    return model

def save_model(model: Model, model_name: str) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{model_name}_model.npz"
    np.savez_compressed(out_path, **model)
