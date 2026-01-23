import numpy as np
import pandas as pd
import src.common as cmn

def solve_run_expectancies(lineup_matrices: list[cmn.Matrix]) -> list[cmn.Vector]:
    n = len(lineup_matrices)
    if n == 0:
        raise ValueError("lineup_matrices must not be empty")
    if any(p.shape != (25, 25) for p in lineup_matrices):
        raise ValueError("Each player matrix must be of shape (25, 25)")

    # 選手ごとに、一時的状態の遷移行列と報酬ベクトルを作成
    q_list: list[cmn.Matrix] = []
    r_list: list[cmn.Vector] = []
    for player_matrix in lineup_matrices:
        q_i = player_matrix[:24, :24]
        r_i = (q_i * cmn.SCORE_MATRIX[:24, :24]).sum(axis=1)
        q_list.append(q_i)
        r_list.append(r_i)

    # 選手を並べて、統合した一時的状態の遷移行列と報酬ベクトルを作成
    q_all: cmn.Matrix = np.zeros((24 * n, 24 * n))
    r_all: cmn.Vector = np.zeros(24 * n)
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
    try:
        run_expectancy = np.linalg.solve(np.eye(24 * n) - q_all, r_all)
    except np.linalg.LinAlgError as e:
        raise ValueError("Failed to solve for run expectancy due to singular matrix.") from e

    # 各選手ごとに分割して返す
    run_expectancy_list: list[cmn.Vector] = np.split(run_expectancy, n)
    return run_expectancy_list

def print_run_expectancies(
    run_expectancies: list[cmn.Vector],
    player_names: list[str] | None = None
) -> None:
    if any(re.shape != (24,) for re in run_expectancies):
        raise ValueError("Each run expectancy vector must be of shape (24,)")

    outs_labels = ["0 Out", "1 Out", "2 Out"]
    base_labels = [cmn.BASE_STR_MAP[i] for i in range(8)]

    print("=== Run Expectancies ===")
    for i, re in enumerate(run_expectancies):
        df = pd.DataFrame(re.reshape(3, 8), index=outs_labels, columns=base_labels)
        player_label = f"{i + 1}. {player_names[i]}" if player_names is not None else f"Player {i + 1}"
        print(f"{player_label} at Bat")
        print(df.round(3))
        print()
