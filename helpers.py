import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

letter_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

def move_probabilities(vals):
    probs = np.array(vals.cpu())
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs**3
    probs = probs / probs.sum()
    return np.nan_to_num(probs, nan=0.0)


def move_to_matrix(move):
    move = move.uci()

    from_output_layer = np.zeros((8, 8))
    from_row = 8 - int(move[1])
    from_column = letter_dict[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter_dict[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer], dtype=np.float32)
