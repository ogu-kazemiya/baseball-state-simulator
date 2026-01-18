import numpy as np
import pandas as pd
from src.common.types import Matrix, Vector
from src.common.constants import BASE_STR_MAP
from src.common.model_rules import SCORE_MATRIX

def solve_run_expectancy(lineup_matrices: list[Matrix]) -> list[Vector]:
    n = len(lineup_matrices)
    if n == 0:
        raise ValueError("lineup_matrices must not be empty")
    if any(p.shape != (25, 25) for p in lineup_matrices):
        raise ValueError("Each player matrix must be of shape (25, 25)")

    # 選手ごとに、一時的状態の遷移行列と報酬ベクトルを作成
    q_list: list[Matrix] = [] # 一時的状態の遷移行列(24,24)のリスト
    r_list: list[Vector] = [] # 報酬ベクトル(24)のリスト
    for player_matrix in lineup_matrices:
        q_i = player_matrix[:24, :24]
        r_i = (q_i * SCORE_MATRIX[:24, :24]).sum(axis=1)
        q_list.append(q_i)
        r_list.append(r_i)

    # 選手を並べて、統合した一時的状態の遷移行列と報酬ベクトルを作成
    q_all: Matrix = np.zeros((24 * n, 24 * n)) # 統合した一時的状態の遷移行列(24n,24n)
    r_all: Vector = np.zeros(24 * n) # 統合した報酬ベクトル(24n)

    if n == 1:
        q_all = q_list[0]
    else:
        for i in range(n):
            next_batter = (i + 1) % n
            row_start = 24 * i
            row_end = 24 * (i + 1)
            col_start = 24 * next_batter
            col_end = 24 * (next_batter + 1)
            q_all[row_start:row_end, col_start:col_end] = q_list[i]
    for i in range(n):
        r_all[24 * i:24 * (i + 1)] = r_list[i]

    # 得点期待値の計算
    # 吸収マルコフ連鎖の基本行列をを M = (I - Q)^(-1) とする
    # 求める期待値ベクトルは E = M R
    # つまり (I - Q) E = R を解く
    run_expectancy = np.linalg.solve(np.eye(24 * n) - q_all, r_all) # 状態別期待値ベクトル(24n)

    # 各選手ごとに分割して返す
    run_expectancy_list: list[Vector] = np.split(run_expectancy, n) # 状態別期待値ベクトル(24)のリスト
    return run_expectancy_list

def print_run_expectancies(run_expectancies: list[Vector]) -> None:
    if any(re.shape != (24,) for re in run_expectancies):
        raise ValueError("Each run expectancy vector must be of shape (24,)")
    
    outs_labels = ["0 Out", "1 Out", "2 Out"]
    base_labels = [BASE_STR_MAP[i] for i in range(8)]

    for i, re in enumerate(run_expectancies):
        df = pd.DataFrame(re.reshape(3, 8), index=outs_labels, columns=base_labels)
        print(f"=== Run Expectancy: Player {i + 1} at Bat ===")
        print(df.round(3))
        print()
