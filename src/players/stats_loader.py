from pathlib import Path
import pandas as pd

# カラム名の表記ゆれを吸収するマッピング辞書
COLUMN_MAPPING = {
    "name": "Name", "player": "Name",
    "pa": "PA",
    "ab": "AB",
    "h": "H",
    "1b": "1B",
    "2b": "2B",
    "3b": "3B",
    "hr": "HR", "homerun": "HR",
    "bb": "BB", "walk": "BB",
    "ibb": "IBB",
    "hbp": "HBP",
    "so": "SO", "k": "SO", "strikeout": "SO",
    "sh": "SH",
    "sf": "SF",
}

def load_stats_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="cp932")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")

    df.columns = [c.strip().lower().replace(" ", "").replace("_", "") for c in df.columns]
    new_columns = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            new_columns[col] = COLUMN_MAPPING[col]
    df = df.rename(columns=new_columns)

    if "Name" not in df.columns:
        df.insert(0, "Name", [f"Player {i+1}" for i in range(len(df))])

    return df
