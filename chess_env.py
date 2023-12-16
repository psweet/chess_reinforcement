from gymnasium import Env
import chess
import numpy as np
import chess.pgn as pgn
import re
from helpers import get_material_balance

from IPython.display import clear_output, display

class ChessEnv(Env):
    def __init__(self, **kwargs) -> None:
        self.reset()

    def reset(self):
        self.game = chess.Board()
        
        observation = self.read_board()
        reward = 0
        ended = False
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
    
    def get_reward(self, player):
        material = self.get_reward_material()
        checkmate = self.get_reward_checkmate()
        captures = self.get_reward_captures()
        if player.color == chess.BLACK:
            material *= -1
            captures *= -1

        not_finished = 0
        if (len(self.game.move_stack) > 50) and (not self.game.is_checkmate()):
            not_finished = -100

        if self.game.can_claim_draw():
            not_finished = -100
    
        return material + captures + checkmate + not_finished
    
    def get_reward_material(self) -> float:
        """Rewards based on material balance"""
        return get_material_balance(self.game)
    
    def get_reward_checkmate(self) -> float:
        """Rewards based on checkmate only"""        
        if self.game.is_checkmate:
            return 100
        
        return 0
    
    def get_reward_captures(self) -> float:
        """Rewards based on captures only"""
        if len(self.game.move_stack) == 0:
            return 0
        
        board = self.game
        current_material = get_material_balance(board)

        move = board.pop()
        last_material = get_material_balance(board)
        
        board.push(move)
        return current_material - last_material
        
    
    def step(self, action, player):
        self.game.push(action)
        observation = self.read_board()
        reward = self.get_reward(player)
        ended = self.game.is_game_over()

        if len(self.game.move_stack) > 50:
            ended = True

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