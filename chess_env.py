from gymnasium import Env
import random
import chess
import numpy as np
import chess.pgn as pgn


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
 
    
    def read_board(self):
        state = pgn.Game.from_board(self.game)
        exporter = pgn.StringExporter(columns=None, headers=False, variations=True, comments=False)
        pgn_string = state.accept(exporter)
        return pgn_string
    
    def get_reward(self):
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
        return np.array([move for move in self.game.legal_moves])