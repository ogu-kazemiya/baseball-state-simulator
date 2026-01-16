import numpy as np
import numpy.typing as npt
import src.common.constants as consts

def _create_score_matrix() -> npt.NDArray[np.int64]:
    score_matrix = np.zeros((25, 25), dtype=np.int64)

    for from_state in range(25):
        for to_state in range(25):
            from_outs =  from_state // 8
            from_runners = consts.BASE_BIT_MAP[from_state % 8].bit_count()
            to_outs = from_state // 8
            to_runners = consts.BASE_BIT_MAP[to_state % 8].bit_count()
            
            score = (from_outs + from_runners + 1) - (to_outs + to_runners)
            if from_outs > to_outs:
                score = -1 # アウトは減らない
            if (from_state % 8 == consts.BASE_BIT_MAP[0b100] and
                to_state % 8 == consts.BASE_BIT_MAP[0b011]):
                score = -1 # ランナーは戻らない
            if to_outs == 3:
                score = min(score, 0) # 3アウト遷移は得点0とする
            if score < 0:
                score = -1 # 不可能な遷移は-1とする

            score_matrix[from_state, to_state] = score
    return score_matrix

SCORE_MATRIX = _create_score_matrix()
