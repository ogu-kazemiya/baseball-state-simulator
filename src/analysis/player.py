import numpy as np
from src.common.types import Model, Matrix
from src.common.matrix_utils import normalize_transition_matrix

def build_player_matrix(transition_model: Model, player_probs: dict[str, float]) -> Matrix:
    if any(result not in player_probs for result in transition_model.keys()):
        raise ValueError("Model does not contain all player result types.")

    player_matrix = np.zeros((25, 25), dtype=np.float64)
    for result, prob in player_probs.items():
        if result not in transition_model:
            continue
        player_matrix += transition_model[result] * prob

    player_matrix = normalize_transition_matrix(player_matrix)
    return player_matrix
