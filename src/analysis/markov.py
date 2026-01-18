import numpy as np
import numpy.typing as npt
import src.common.types as types
import src.common.model_rules as model_rules

def calculate_re24(player_matrices: list[types.Matrix]) -> list[types.Vector]:
    n = len(player_matrices)
    if n == 0:
        raise ValueError("player_matrices must not be empty")
    if any(p.shape != (25, 25) for p in player_matrices):
        raise ValueError("Each player matrix must be of shape (25, 25)")

    # 選手ごとに、一時的状態の遷移行列と報酬ベクトルを作成
    q_list: list[types.Matrix] = [] # 一時的状態の遷移行列(24,24)のリスト
    r_list: list[types.Vector] = [] # 報酬ベクトル(24)のリスト

    for p in player_matrices:
        q_i = p[:24, :24]
        r_i = (q_i * model_rules.SCORE_MATRIX[:24, :24]).sum(axis=1)
        q_list.append(q_i)
        r_list.append(r_i)

    # 選手を並べて、統合した一時的状態の遷移行列と報酬ベクトルを作成
    q_all: types.Matrix = np.zeros((24 * n, 24 * n)) # 統合した一時的状態の遷移行列(24n,24n)
    r_all: types.Vector = np.zeros(24 * n) # 統合した報酬ベクトル(24n)

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
    re24_list: list[types.Vector] = np.split(run_expectancy, n) # 状態別期待値ベクトル(24)のリスト
    return re24_list
