from typing import Literal
import warnings
import pandas as pd
import src.common.constants as consts
import src.common.model_rules as model_rules

def build_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(_add_state_column)
          .pipe(_add_next_state_column)
          .pipe(_add_state_str_column)
          .pipe(_add_next_state_str_column)
          .pipe(_validate_states_transition, mode="drop")
    )

def _add_state_column(df: pd.DataFrame) -> pd.DataFrame:
    base_bit_map_inv: dict[int, int] = {v: k for k, v in consts.BASE_BIT_MAP}
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

def _add_next_state_column(df: pd.DataFrame) -> pd.DataFrame:
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

def _add_state_str_column(df: pd.DataFrame) -> pd.DataFrame:
    df["state_str"] = df["state"].map(consts.STATE_STR_MAP)
    return df

def _add_next_state_str_column(df: pd.DataFrame) -> pd.DataFrame:
    df["next_state_str"] = df["next_state"].map(consts.STATE_STR_MAP)
    return df

def _validate_states_transition(
    df: pd.DataFrame,
    mode: Literal["raise", "warn", "ignore", "drop", "return"] = "raise"
) -> pd.DataFrame:
    estimated_scores_arr = model_rules.SCORE_MATRIX[df["state"].values, df["next_state"].values]
    estimated_scores = pd.Series(estimated_scores_arr, index=df.index).astype(int)
    estimated_scores = estimated_scores.where(df["events"].isin(consts.PA_EVENTS), estimated_scores - 1) # 打席が未完了
    actual_scores = (df["post_bat_score"] - df["bat_score"]).astype(int)

    is_valid_transition = (
        (estimated_scores == actual_scores)
        | (df["next_state"] == 24) # 3アウト
        | (df["next_state"] == -1) # サヨナラ
    )
    is_invalid_transition = ~is_valid_transition

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
