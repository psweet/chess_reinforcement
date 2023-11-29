import chess.pgn as pgn
from chess_gif.gif_maker import Gifmaker

game = pgn.Game().from_board(board)
game.headers["Event"] = "Reinforcement Learning MSML Final Project"
exporter = pgn.StringExporter(columns=None, headers=False, variations=True, comments=False)
pgn_string = game.accept(exporter)

obj = Gifmaker()
obj.make_gif_from_pgn_string(pgn_string=pgn_string, gif_file_path='chess_game.gif')

