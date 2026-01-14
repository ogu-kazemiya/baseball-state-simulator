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
