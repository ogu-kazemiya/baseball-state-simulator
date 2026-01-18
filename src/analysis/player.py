import numpy as np
import numpy.typing as npt
import src.common.types as types
import src.common.constants as consts
import src.common.matrix_utils as matrix_utils

def create_player_matrix(model: types.Model, player_probs: dict[str, float]) -> npt.NDArray[np.float64]:
    player_matrix = np.zeros((25, 25), dtype=np.float64)

    for result, prob in player_probs.items():
        if result not in model:
            continue
        player_matrix += model[result] * prob

    player_matrix = matrix_utils.normalize_transition_matrix(player_matrix)
    return player_matrix
