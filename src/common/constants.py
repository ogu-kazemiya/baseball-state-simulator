# statcastのcolumns
REQUIRED_COLS = [
    "game_date", "home_team", "away_team", "game_type", # 試合情報
    "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number", # イニング・打席
    "outs_when_up", "balls", "strikes", "on_1b", "on_2b", "on_3b", # カウント・ランナー
    "bat_score", "post_bat_score", "fld_score", "post_home_score", "post_away_score", # スコア
    "events", "description", # 打撃結果
]

# イベントの分類
HIT_EVENTS = ["single", "double", "triple", "home_run"]
ON_BASE_EVENTS = ["walk", "intent_walk", "hit_by_pitch", "catcher_interf"]
STRIKEOUT_EVENTS = ["strikeout", "strikeout_double_play"]
FIELD_OUT_EVENTS = [
    "field_out", "force_out", "fielders_choice_out", "grounded_into_double_play", "double_play", "triple_play",
    "field_error", "fielders_choice", "sac_bunt_double_play", "sac_fly", "sac_fly_double_play"
]
SAC_BUNT_EVENTS = ["sac_bunt"]
EXCLUDE_EVENTS = ["truncated_pa", "ejection", "game_advisory"]
PA_EVENTS = HIT_EVENTS + ON_BASE_EVENTS + STRIKEOUT_EVENTS + FIELD_OUT_EVENTS + SAC_BUNT_EVENTS
ALL_EVENTS = PA_EVENTS + EXCLUDE_EVENTS

# 塁状況のバイナリ表現マッピング
BASE_BIT_MAP = {
    0: 0b000, 1: 0b001, 2: 0b010, 3: 0b100,
    4: 0b011, 5: 0b101, 6: 0b110, 7: 0b111,
}

# 塁状況の文字列表現マッピング
BASE_STR_MAP = {
    0: "___", 1: "1__", 2: "_2_", 3: "__3",
    4: "12_", 5: "1_3", 6: "_23", 7: "123",
}

# stateの文字列表現マッピング
STATE_STR_MAP = {
    0: "0/___", 1: "0/1__", 2: "0/_2_", 3: "0/__3",
    4: "0/12_", 5: "0/1_3", 6: "0/_23", 7: "0/123",
    8: "1/___", 9: "1/1__", 10: "1/_2_", 11: "1/__3",
    12: "1/12_", 13: "1/1_3", 14: "1/_23", 15: "1/123",
    16: "2/___", 17: "2/1__", 18: "2/_2_", 19: "2/__3",
    20: "2/12_", 21: "2/1_3", 22: "2/_23", 23: "2/123",
    24: "3/___",
    -1: "Excl"
}
