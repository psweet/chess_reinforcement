from gymnasium import Env
import random
import chess
import numpy as np
import chess.pgn as pgn
import cairosvg
import io
from PIL import Image

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


    def read_board(self):
        file_name = "current_game.svg"
        svg = chess.svg.board(self.game)
        outputfile = open(file_name, "w")
        outputfile.write(svg)
        outputfile.close()
        mem = io.BytesIO()
        cairosvg.svg2png(url=file_name, write_to=mem)
        pixels = np.array(Image.open(mem))[:,:,0]

        # import matplotlib.pyplot as plt
        # plt.imshow(pixels)
        # plt.show()

        return pixels
    
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