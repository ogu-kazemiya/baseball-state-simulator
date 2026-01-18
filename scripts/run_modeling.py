# %%
# 1. セットアップ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.modeling.statcast_loader import load_statcast
from src.modeling.state import assign_state_features
from src.modeling.transition import aggregate_count_matrices, build_model, save_model

# %%
# 2. データ読み込み
start_year = 2016
end_year = 2025

df = load_statcast(start_year, end_year)
print(f"DataFrame shape: {df.shape}")

# %%
# 3. stateカラムの構築
df = assign_state_features(df)
print(f"DataFrame shape: {df.shape}")

# %%
# 4. 遷移回数行列の計算
count_matrices = aggregate_count_matrices(df)

# %% 
# 5. モデルの作成
model = build_model(count_matrices)
save_model(model, model_name="main")
