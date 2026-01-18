import numpy as np
import numpy.typing as npt
from src.common.types import Matrix
from src.common.model_rules import SCORE_MATRIX

def simulate_states(
    lineup_matrices: list[Matrix],
    initial_batter_index: int = 0,
    initial_state: int = 0,
    num_simulations: int = 100000,
) -> npt.NDArray[np.int64]:
    n_batters = len(lineup_matrices)
    if n_batters == 0:
        raise ValueError("lineup_matrices must not be empty")
    if any(p.shape != (25, 25) for p in lineup_matrices):
        raise ValueError("Each player matrix must be of shape (25, 25)")
    if not (0 <= initial_batter_index < n_batters):
        raise ValueError("initial_batter_index must be between 0 and number of players - 1")
    if not (0 <= initial_state < 24):
        raise ValueError("initial_state must be between 0 and 23")

    stacked_matrix = np.stack(lineup_matrices) # 遷移行列のスタック(n_batters, 25, 25)
    current_batters = np.full(num_simulations, initial_batter_index, dtype=np.int64)
    current_states = np.full(num_simulations, initial_state, dtype=np.int64)
    total_runs = np.zeros(num_simulations, dtype=np.int64)
    active_mask = np.ones(num_simulations, dtype=bool)

    while np.any(active_mask):
        # 未完了の試行の状態と打者を抽出
        active_states = current_states[active_mask]
        active_batters = current_batters[active_mask]
        n_active = active_states.shape[0]

        # 遷移確率に基づき次の状態を決定
        transition_probs = stacked_matrix[active_batters, active_states, :]
        cumulative_probs = np.cumsum(transition_probs, axis=1)
        random_values = np.random.rand(n_active, 1)
        next_states = (cumulative_probs < random_values).sum(axis=1)
        next_states = np.minimum(next_states, 24) # 小数点誤差対策

        # 得点の更新
        step_scores = SCORE_MATRIX[active_states, next_states]
        total_runs[active_mask] += step_scores

        # 状態と打者の更新
        current_batters[active_mask] = (active_batters + 1) % n_batters
        current_states[active_mask] = next_states

        # 完了した試行のマスク更新
        finished_mask = (next_states == 24)
        active_indices = np.where(active_mask)[0]
        active_mask[active_indices[finished_mask]] = False
    
    return total_runs

def calculate_prob_at_least(runs_array: npt.NDArray[np.int64], target_score: int) -> float:
    if runs_array.size == 0:
        raise ValueError("runs_array must not be empty")
    if target_score < 0:
        raise ValueError("target_score must be non-negative")

    prob = np.mean(runs_array >= target_score)
    return prob

def calculate_score_distribution(runs_array: npt.NDArray[np.int64]) -> dict[int, float]:
    if runs_array.size == 0:
        raise ValueError("runs_array must not be empty")

    unique, counts = np.unique(runs_array, return_counts=True)
    total = runs_array.size
    distribution = {int(k): v / total for k, v in zip(unique, counts)}
    return distribution

def print_simulation_report(runs_array: npt.NDArray[np.int64]) -> None:
    if runs_array.size == 0:
        raise ValueError("runs_array must not be empty")

    mean = np.mean(runs_array)
    std_dev = np.std(runs_array)
    dist = calculate_score_distribution(runs_array)

    print("=== Simulation Report ===")
    print(f" Trials: {len(runs_array):,}")
    print(f" Mean  : {mean:.3f}")
    print(f" StdDev: {std_dev:.3f}")
    print("-" * 25)
    print(" [Key Probabilities]")
    print(f"  Score >= 1: {calculate_prob_at_least(runs_array, 1):.1%}")
    print(f"  Score >= 2: {calculate_prob_at_least(runs_array, 2):.1%}")
    print(f"  Score >= 3: {calculate_prob_at_least(runs_array, 3):.1%}")
    print(f"  Score >= 4: {calculate_prob_at_least(runs_array, 4):.1%}")
    print("-" * 25)
    print(" [Distribution]")
    for score in sorted(dist.keys()):
        prob = dist[score]
        if prob < 0.001: continue # 0.1%未満は省略
        bar = "#" * int(prob * 40) # 簡易グラフ
        print(f"  {score:2d} runs: {prob:6.1%} |{bar}")
    print("=========================")
    print()
