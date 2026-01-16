# %%
# 1. セットアップ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.estimation as est

# %%
# 2. データ読み込み
start_year = 2016
end_year = 2025

df = est.get_statcast(start_year, end_year)
print(f"DataFrame shape: {df.shape}")

# %%
# 3. stateカラムの構築
df = est.build_state_columns(df)
print(f"DataFrame shape: {df.shape}")

# %%
# 4. 遷移回数行列の計算
count_matrices = est.compute_count_matrices(df)
est.save_count_matrices(count_matrices)
