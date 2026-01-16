import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from pybaseball import statcast, cache
import src.common.constants as consts

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATCAST_DATA_DIR = PROJECT_ROOT / "data" / "statcast"

def get_statcast(
    start_year: int,
    end_year: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if end_year is None:
        end_year = start_year
    columns = list(dict.fromkeys(columns)) if columns is not None else consts.REQUIRED_COLS
    years = range(start_year, end_year + 1)

    for year in years:
        _ensure_season_statcast(year)

    df_list = []
    for year in years:
        df_year = _get_season_statcast(year, columns)
        df_list.append(df_year)

    return pd.concat(df_list, ignore_index=True)

def _get_season_statcast(year: int, columns: list[str]) -> pd.DataFrame:
    file_path = STATCAST_DATA_DIR / f"statcast_{year}.parquet"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Statcast data for year {year} not found in cache.")

    df = pd.read_parquet(file_path, columns=columns)
    _validate_columns(df, columns)
    return df

def _ensure_season_statcast(year: int) -> None:
    os.makedirs(STATCAST_DATA_DIR, exist_ok=True)
    file_path = STATCAST_DATA_DIR / f"statcast_{year}.parquet"

    if os.path.exists(file_path):
        return

    cache.enable()
    warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")
    months = range(3, 12)
    df_list = []

    for month in tqdm(months, desc=f"Downloading Statcast data for {year}"):
        start_date = datetime(year, month, 1)
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        str_start_date = start_date.strftime("%Y-%m-%d")
        str_end_date = end_date.strftime("%Y-%m-%d")

        df_month = statcast(start_dt=str_start_date, end_dt=str_end_date, verbose=False)
        df_list.append(df_month)
        time.sleep(1)

    df = pd.concat(df_list, ignore_index=True)
    df.to_parquet(file_path)

def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    required_columns = set(required_columns)
    actual_columns = set(df.columns)
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
