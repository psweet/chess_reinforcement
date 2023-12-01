from gymnasium import Env
import random
import chess
import numpy as np
import chess.pgn as pgn
import re

from IPython.display import clear_output, display

class ChessEnv(Env):
    def __init__(self, **kwargs) -> None:
        self.reset()

    def reset(self):
        self.game = chess.Board()
        self.color = random.choice([chess.WHITE, chess.BLACK])
        self.modifier = 1
        if self.color == chess.BLACK:
            self.modifier = -1
        
        observation = self.read_board()
        reward = self.get_reward()
        ended = self.game.is_game_over()
        info = {
            "Move made" : "None",
            "Reward" : reward,
            "Ended" : ended,
            "Observed": observation
        }
        return observation, info
 
    def pgn(self):
        game = pgn.Game.from_board(self.game)
        exporter = pgn.StringExporter(columns=None, headers=False, variations=True, comments=False)
        game_string = game.accept(exporter)
        return game_string
    
    def move_2_rep(self, move, board):
        self.game.push_san(move).uci()
        move = str(board.pop())

        letter_2_num = {
            "a":0,
            "b":1,
            "c":2,
            "d":3,
            "e":4,
            "f":5,
            "g":6,
            "h":7
        }

        num_2_letter = {
            0:"a",
            1:"b",
            2:"c",
            3:"d",
            4:"e",
            5:"f",
            6:"g",
            7:"h"
        }
        
        from_output_layer = np.zeroes((8, 8))
        from_row = 8 - int(move[1])
        from_column = letter_2_num[move[0]]
        from_output_layer[from_row, from_column] = 1

        to_output_layer = np.zeroes((8, 8))
        to_row = 8 - int(move[3])
        to_column = letter_2_num[move[2]]
        to_output_layer[to_row, to_column] = 1

        return np.stack([from_output_layer, to_output_layer], dtype=np.float32)

        


    def read_board(self):
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        layers = []

        board = str(self.game)
        for piece in pieces:
            s = re.sub(f'[^{piece}{piece.upper()} \n]', '.', board)
            s = re.sub(f'{piece}', '-1', s)
            s = re.sub(f'{piece.upper()}', '1', s)
            s = re.sub('\.', '0', s)

            board_matrix = []
            for row in s.split("\n"):
                row = row.split(" ")
                row = [int(x) for x in row]
                board_matrix.append(row)

            layers.append(board_matrix)

        board_state = np.stack(layers)

        return board_state
    
    def get_reward(self) -> float:
        board = self.game

        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        return (
            chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
            3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
            3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
            5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
            9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
        ) * self.modifier
    
    def step(self, action):
        self.game.push(action)
        observation = self.read_board()
        reward = self.get_reward()
        ended = self.game.is_game_over()
        info = {
            "Move made" : action,
            "Reward" : reward,
            "Ended" : ended,
            "Observed": observation
        }
        return observation, reward, ended, False, info
    
    def render(self):
        clear_output(wait=True)
        return display(self.game)
    
    @property
    def action_space(self):
        return [move for move in self.game.legal_moves]