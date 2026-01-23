from typing import Literal
import warnings
import pandas as pd
from src.common.constants import BASE_BIT_MAP, STATE_STR_MAP, PA_EVENTS
from src.common.model_rules import SCORE_MATRIX

def assign_state_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(_assign_state)
          .pipe(_assign_next_state)
          .pipe(_assign_state_str)
          .pipe(_assign_next_state_str)
          .pipe(_validate_transition, mode="drop")
    )

def _assign_state(df: pd.DataFrame) -> pd.DataFrame:
    base_bit_map_inv: dict[int, int] = {v: k for k, v in BASE_BIT_MAP.items()}
    outs = df["outs_when_up"].astype(int)
    base_bits = (
        (df["on_1b"].notna().astype(int) * 1)
        + (df["on_2b"].notna().astype(int) * 2)
        + (df["on_3b"].notna().astype(int) * 4)
    )
    base = base_bits.map(base_bit_map_inv).astype(int)
    state = outs * 8 + base
    df["state"] = state.astype(int)
    return df

def _assign_next_state(df: pd.DataFrame) -> pd.DataFrame:
    df = _sort_statcast_df(df)

    next_game_pk = df["game_pk"].shift(-1)
    next_inning = df["inning"].shift(-1)
    next_inning_topbot = df["inning_topbot"].shift(-1)

    is_game_end = (next_game_pk != df["game_pk"]) | next_game_pk.isna()
    is_inning_end = (
        is_game_end
        | (next_inning != df["inning"])
        | (next_inning_topbot != df["inning_topbot"])
    )
    is_walkoff = (
        is_game_end
        & (df["inning_topbot"] == "Bot")
        & (df["post_home_score"] > df["post_away_score"])
    )

    next_state = df["state"].shift(-1)
    next_state = next_state.mask(is_inning_end & ~is_walkoff, 24) # 3アウト
    next_state = next_state.mask(is_walkoff, -1)  # サヨナラ

    df["next_state"] = next_state.fillna(-1).astype(int)
    return df

def _assign_state_str(df: pd.DataFrame) -> pd.DataFrame:
    df["state_str"] = df["state"].map(STATE_STR_MAP)
    return df

def _assign_next_state_str(df: pd.DataFrame) -> pd.DataFrame:
    df["next_state_str"] = df["next_state"].map(STATE_STR_MAP)
    return df

def _validate_transition(
    df: pd.DataFrame,
    mode: Literal["raise", "warn", "ignore", "drop", "return"] = "raise"
) -> pd.DataFrame:
    estimated_scores_arr = SCORE_MATRIX[df["state"].values, df["next_state"].values]
    estimated_scores = pd.Series(estimated_scores_arr, index=df.index).astype(int)
    estimated_scores = estimated_scores.where(df["events"].isin(PA_EVENTS), estimated_scores - 1)
    actual_scores = (df["post_bat_score"] - df["bat_score"]).astype(int)

    is_valid_transition = (
        (estimated_scores == actual_scores)
        | (df["next_state"] == 24) # 3アウト
        | (df["next_state"] == -1) # サヨナラ
    )
    is_invalid_transition = ~is_valid_transition

    df["estimated_score"] = estimated_scores
    df["actual_score"] = actual_scores

    if mode == "raise":
        if is_invalid_transition.any():
            invalid_count = is_invalid_transition.sum()
            raise ValueError(f"Found {invalid_count} invalid state transitions.")
    if mode == "warn":
        if is_invalid_transition.any():
            invalid_count = is_invalid_transition.sum()
            warnings.warn(f"Warning: Found {invalid_count} invalid state transitions.")
    if mode == "drop":
        df = df[is_valid_transition].reset_index(drop=True)
    if mode == "return":
        df = df[is_invalid_transition].reset_index(drop=True)
    return df

def _sort_statcast_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["game_pk", "at_bat_number", "pitch_number"]
    return df.sort_values(by=sort_cols).reset_index(drop=True)
