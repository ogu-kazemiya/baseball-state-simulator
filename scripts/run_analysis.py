# %%
# 1. セットアップ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import src.common as cmn
import src.players as pl
import src.analysis as ana

# %%
# 2. データ読み込み
model: cmn.Model = np.load(PROJECT_ROOT / "data" / "artifacts" / "main_model.npz")
stats_df = pl.load_stats_csv(PROJECT_ROOT / "data" / "examples" / "stats_2025_LAD.csv")

# %%
# 3. 選手行列の作成
batting_order = [
    "Shohei Ohtani", "Mookie Betts", "Freddie Freeman",
    "Will Smith", "Teoscar Hernández", "Andy Pages",
    "Max Muncy", "Michael Conforto", "Tommy Edman"
]

lineup_df = pl.pick_lineup(stats_df, batting_order)
lineup_probs = pl.convert_stats_to_probs(lineup_df)
lineup_matrices = pl.build_lineup_matrices(model, lineup_probs)

# ラインナップを表示
print("=== Lineup ===")
print(pl.get_formatted_stats(lineup_df))
print()

# %%
# 4. 得点期待値の計算
run_expectancies = ana.solve_run_expectancies(lineup_matrices)
ana.print_run_expectancies(run_expectancies, player_names=batting_order)

# %%
# 5. モンテカルロシミュレーション
runs_array = ana.simulate_states(lineup_matrices, batter_index=0, state="0/___")
ana.print_simulation_report(runs_array, batter=0, state="0/___")

# %%
# 6. 送りバントの検証
# 2025年阪神タイガースを例に
stats_df_T = pl.load_stats_csv(PROJECT_ROOT / "data" / "examples" / "stats_2025_T.csv")
batting_order_T = ["Chikamoto", "Nakano", "Morishita", "Sato", "Ohyama", "Obata", "Sakamoto", "Maegawa", "Murakami"]
lineup_df_T = pl.pick_lineup(stats_df_T, batting_order_T)
lineup_probs_T = pl.convert_stats_to_probs(lineup_df_T)
lineup_matrices_T = pl.build_lineup_matrices(model, lineup_probs_T)

# 2番中野 0/1__ vs 3番森下 1/_2_
runs_array_hitting = ana.simulate_states(lineup_matrices_T, batter_index=1, state="0/1__")
ana.print_simulation_report(runs_array_hitting, batter="中野", state="0/1__")
runs_array_bunting = ana.simulate_states(lineup_matrices_T, batter_index=2, state="1/_2_")
ana.print_simulation_report(runs_array_bunting, batter="森下", state="1/_2_")
