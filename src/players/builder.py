import numpy as np
import pandas as pd
from src.common.types import Model, Matrix
from src.common.matrix_utils import normalize_transition_matrix
from src.players.stats_utils import validate_and_fill_stats

def convert_stats_to_probs(lineup_stats: pd.DataFrame) -> pd.DataFrame:
    stats = validate_and_fill_stats(lineup_stats)

    # 打席結果の集計
    denominator = stats["PA"] - stats["IBB"] - stats["SH"]
    singles = stats["H"] - stats["2B"] - stats["3B"] - stats["HR"]
    doubles = stats["2B"]
    triples = stats["3B"]
    home_runs = stats["HR"]
    walks = (stats["BB"] - stats["IBB"]) + stats["HBP"]
    strikeouts = stats["SO"]
    field_outs = stats["PA"] - stats["H"] - stats["BB"] - stats["HBP"] - stats["SO"] - stats["SH"]

    if (denominator <= 0).any():
        invalid_rows = stats[denominator <= 0]
        if "Name" in stats.columns:
            player_names = invalid_rows["Name"].tolist()
            msg_target = f"Players: {player_names}"
        else:
            msg_target = f"Indices: {invalid_rows.index.tolist()}"
        raise ValueError(f"Effective PA (PA - IBB - SH) must be > 0. {msg_target}")

    # 確率の計算
    probs = pd.DataFrame(index=stats.index)
    probs["single"] = singles / denominator
    probs["double"] = doubles / denominator
    probs["triple"] = triples / denominator
    probs["home_run"] = home_runs / denominator
    probs["walk"] = walks / denominator
    probs["strikeout"] = strikeouts / denominator
    probs["field_out"] = field_outs / denominator

    if "Name" in stats.columns:
        probs["Name"] = stats["Name"]

    return probs

def build_lineup_matrices(transition_model: Model, lineup_probs: pd.DataFrame) -> list[Matrix]:
    model_results = set(transition_model.keys())
    probs_results = set(lineup_probs.columns)
    missing_results = model_results - probs_results
    if missing_results:
        raise ValueError(f"Lineup probabilities are missing results: {missing_results}")

    lineup_matrices = []
    for idx, row in lineup_probs.iterrows():
        player_matrix = np.zeros((25, 25), dtype=np.float64)
        for result, transition_matrix in transition_model.items():
            prob = row[result]
            player_matrix += transition_matrix * prob
        try:
            player_matrix = normalize_transition_matrix(player_matrix)
        except Exception as e:
            player_name = row.get("Name", f"Batter {idx}")
            raise ValueError(f"Error normalizing player matrix for {player_name}") from e
        lineup_matrices.append(player_matrix)

    return lineup_matrices
