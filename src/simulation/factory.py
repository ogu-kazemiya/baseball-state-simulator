import numpy as np
import numpy.typing as npt
import src.common.constants as consts
import src.common.matrix_utils as matrix_utils

Model = dict[str, npt.NDArray[np.float64]]

def create_model(count_matrices: dict[str, npt.NDArray[np.int64]]) -> Model:
    model: Model = {}

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

def create_player_matrix(model: Model, player_probs: dict[str, float]) -> npt.NDArray[np.float64]:
    player_matrix = np.zeros((25, 25), dtype=np.float64)

    for result, prob in player_probs.items():
        if result not in model:
            continue
        player_matrix += model[result] * prob

    player_matrix = matrix_utils.normalize_transition_matrix(player_matrix)
    return player_matrix
