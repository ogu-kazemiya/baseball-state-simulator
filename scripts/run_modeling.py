# %%
# 1. セットアップ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.modeling as mdl

# %%
# 2. データ読み込み
start_year = 2016
end_year = 2025

df = mdl.get_statcast(start_year, end_year)
print(f"DataFrame shape: {df.shape}")

# %%
# 3. stateカラムの構築
df = mdl.build_state_columns(df)
print(f"DataFrame shape: {df.shape}")

# %%
# 4. 遷移回数行列の計算
count_matrices = mdl.compute_count_matrices(df)
mdl.save_count_matrices(count_matrices)

# %% 
# 5. モデルの作成
model = mdl.create_model(count_matrices)
mdl.save_model(model, model_name="main")
