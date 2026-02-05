# %%
# 1. セットアップ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.common as cmn
import src.models as mdl

# %%
# 2. データ読み込み
start_year = 2023
end_year = 2025

df = mdl.load_statcast(start_year, end_year)
print(f"DataFrame shape: {df.shape}")

# %%
# 3. stateカラムの構築
df = mdl.assign_state_features(df)
print(f"DataFrame shape: {df.shape}")

# %%
# 4. 遷移回数行列の計算
count_matrices = mdl.aggregate_count_matrices(df)
cmn.print_matrix_formatted(
    count_matrices["single"],
    title="State Transition Count Matrix (Single)",
    mode="count"
)

# %% 
# 5. モデルの作成
model = mdl.build_model(count_matrices)
mdl.save_model(model, model_name="main")
cmn.print_matrix_formatted(
    model["single"],
    title="State Transition Matrix (Single)",
    mode="rate"
)
