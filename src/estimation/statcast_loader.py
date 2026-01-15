import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pybaseball import statcast, cache
import src.common.constants as consts
import src.estimation.utils as utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATCAST_DATA_DIR = PROJECT_ROOT / "data" / "statcast"

# 指定した期間のstatcastデータを取得
def get_statcast(
    start_year: int,
    end_year: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if end_year is None:
        end_year = start_year
    columns = list(dict.fromkeys(columns)) if columns is not None else consts.REQUIRED_COLS

    df_list = []
    years = range(start_year, end_year + 1)
    for year in tqdm(years, desc="Loading Seasons"):
        df_year = get_season_statcast(year, columns=columns)
        df_list.append(df_year)

    return pd.concat(df_list, ignore_index=True)

# 指定した年のstatcastデータを取得
def get_season_statcast(year: int, columns: list[str] | None = None) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns)) if columns is not None else consts.REQUIRED_COLS

    os.makedirs(STATCAST_DATA_DIR, exist_ok=True)
    file_path = STATCAST_DATA_DIR / f"statcast_{year}.parquet"

    # キャッシュが存在すれば読み込み
    if os.path.exists(file_path):
        try:
           return pd.read_parquet(file_path, columns=columns)
        except Exception as e:
           df = pd.read_parquet(file_path)
           utils.validate_columns(df, columns or [], context=f"get_season_statcast({year}) cache")
           raise e

    # データを取得して保存
    cache.enable()
    start_date = f"{year}-03-01"
    end_date = f"{year}-11-30"
    df = statcast(start_dt=start_date, end_dt=end_date)
    df.to_parquet(file_path)

    utils.validate_columns(df, columns or [], context=f"get_season_statcast({year}) fetch")
    if columns is not None:
        df = df[columns]
    return df
