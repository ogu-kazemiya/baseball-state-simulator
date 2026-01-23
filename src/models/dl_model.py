import numpy as np
from src.common.types import Model
from src.common.model_rules import RESULT_MAPPING
from src.common.constants import BASE_BIT_MAP

def create_dl_model() -> Model:
    results = RESULT_MAPPING.keys()
    hit_bases = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    base_bit_map_inv = {v: k for k, v in BASE_BIT_MAP.items()}

    model: Model = {}
    for result in results:
        transition_matrix = np.zeros((25, 25), dtype=np.float64)
        for from_state in range(25):
            from_outs = from_state // 8
            from_base = from_state % 8
            from_base_bit = BASE_BIT_MAP[from_base]

            # 進塁の処理
            if result in hit_bases:
                base_mask = 0b111 if result != "single" else 0b011 # 単打なら2塁ランナー生還
                bases = hit_bases[result]
                to_base_bit = ((from_base_bit << bases) | (1 << (bases - 1))) & base_mask
            elif result == "walk":
                forced_map = {
                    0b000: 0b001, 0b001: 0b011, 0b010: 0b011, 0b100: 0b101,
                    0b011: 0b111, 0b101: 0b111, 0b110: 0b111, 0b111: 0b111,
                }
                to_base_bit = forced_map[from_base_bit]
            else:  # strikeout, field_out
                to_base_bit = from_base_bit

            to_outs = from_outs + 1 if result in {"strikeout", "field_out"} else from_outs
            to_base = base_bit_map_inv[to_base_bit]
            to_state = min(to_outs * 8 + to_base, 24)  # 3アウトは24
            transition_matrix[from_state, to_state] = 1.0
        model[result] = transition_matrix

    return model
