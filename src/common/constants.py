import numpy as np

# statcastのcolumns
REQUIRED_COLS = [
    "game_date", "home_team", "away_team", "game_type", # 試合情報
    "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number", # イニング・打席
    "outs_when_up", "balls", "strikes", "on_1b", "on_2b", "on_3b", # カウント・ランナー
    "bat_score", "post_bat_score", "fld_score", # スコア
    "events", "description", # 打撃結果
]
