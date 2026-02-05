import numpy as np
import pandas as pd

REQUIRED_COLS = ["PA", "H", "2B", "3B", "HR", "BB", "SO"]
OPTIONAL_COLS = ["IBB", "HBP", "SH", "SF"]

RATE_STATS = {"AVG", "SLG", "OBP", "OPS", "ISO", "BABIP"}
INT_STATS = {"PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "SO", "SH", "SF"}
OTHER_STATS = {"BB/K"}

def pick_lineup(stats_df: pd.DataFrame, batting_order: list[int | str]) -> pd.DataFrame:
    pool = stats_df.copy()

    picked_rows = []
    missing_items = []
    for selector in batting_order:
        if isinstance(selector, int):
            if 0 <= selector < len(pool):
                picked_rows.append(pool.iloc[selector])
            else:
                missing_items.append(f"Index {selector} (out of bounds)")
        elif isinstance(selector, str):
            matched = pool[pool["Name"].str.contains(selector, na=False)] # 部分一致検索
            if len(matched) == 1:
                picked_rows.append(matched.iloc[0])
            elif len(matched) == 0:
                missing_items.append(f"Name '{selector}' (not found)")
            else:
                raise ValueError(f"Name '{selector}' (multiple matches found)")
        else:
            raise ValueError(f"Invalid selector type: {selector} (must be int or str)")

    if missing_items:
        raise ValueError(f"Could not find the following in the stats DataFrame: {missing_items}")
    
    lineup_df = pd.DataFrame(picked_rows).reset_index(drop=True)
    return lineup_df

def validate_and_fill_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    stats = stats_df.copy()

    # 必須カラムの確認
    missing_cols = [col for col in REQUIRED_COLS if col not in stats.columns]
    if missing_cols:
        raise ValueError(f"Missing required stats columns: {missing_cols}")
    if stats[REQUIRED_COLS].isna().any().any():
        bad_rows = stats[stats[REQUIRED_COLS].isna().any(axis=1)]
        bad_names = bad_rows.get("Name", bad_rows.index).tolist()
        raise ValueError(f"NaN values found in required stats columns for: {bad_names}")

    # 任意カラムの補完
    for col in OPTIONAL_COLS:
        if col not in stats.columns:
            stats[col] = 0
        else:
            stats[col] = stats[col].fillna(0)

    return stats

def get_formatted_stats(stats_df: pd.DataFrame, display_stats: list[str] | None = None) -> pd.DataFrame:
    df = validate_and_fill_stats(stats_df)
    if display_stats is None:
        display_stats = ["AVG", "SLG", "OBP", "OPS", "AB", "H", "HR", "BB", "SO"]

    # 指標の計算
    if "AB" not in df.columns:
        df["AB"] = df["PA"] - df["BB"] - df["HBP"] - df["SH"] - df["SF"]
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["AVG"] = df["H"] / df["AB"].replace(0, np.nan)
    df["SLG"] = (df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]) / df["AB"].replace(0, np.nan)
    df["ISO"] = df["SLG"] - df["AVG"]
    df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / df["PA"].replace(0, np.nan)
    df["OPS"] = (df["OBP"] + df["SLG"]).replace(np.nan, 0)
    df["BABIP"] = (df["H"] - df["HR"]) / (df["AB"] - df["SO"] - df["HR"] + df["SF"]).replace(0, np.nan)
    df["BB/K"] = df["BB"] / df["SO"].replace(0, np.nan)

    display_df = pd.DataFrame()
    if "Name" in df.columns:
        display_df["Name"] = df["Name"]
    else:
        display_df.index = df.index

    for stat in display_stats:
        if stat in RATE_STATS:
            display_df[stat] = df[stat].map(_format_rate)
        elif stat in INT_STATS:
            display_df[stat] = df[stat].map(_format_int)
        elif stat in df.columns:
            display_df[stat] = df[stat]
        else:
            raise ValueError(f"Stat '{stat}' not found in stats DataFrame.")

    return display_df

def _format_rate(val: float) -> str:
    if pd.isna(val):
        return ".---"

    s = f"{val:.3f}"
    if s.startswith("0"):
        s = s[1:]
    return s

def _format_int(val: float) -> str:
    if pd.isna(val):
        return "-"
    return str(int(val))
