import pandas as pd

def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    context: str = "check"
) -> None:
    required_columns = set(required_columns)
    actual_columns = set(df.columns)
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"[{context}] Missing columns: {', '.join(missing_columns)}")

def sort_statcast_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["game_pk", "at_bat_number", "pitch_number"]
    return df.sort_values(by=sort_cols).reset_index(drop=True)
