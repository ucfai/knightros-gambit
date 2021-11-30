'''This file contains classes that wrap players of different types.

This class will wrap the custom AI we build, but for now we use it to get moves from stockfish.
'''
from stockfish import Stockfish

class StockfishPlayer:
    '''"AI" class that is a simple wrapper around the Stockfish engine.
    '''
    def __init__(self, operating_system, elo_rating=1300):
        operating_system = operating_system.lower()
        if operating_system == "darwin":
            stockfish_path = "/usr/local/bin/stockfish"
        elif operating_system == "raspi":
            stockfish_path = "n/a"
        elif operating_system == "linux":
            stockfish_path = "../../chess-engine/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"
        elif operating_system == "windows":
            stockfish_path = "n/a"
        else:
            raise ValueError("Operating system must be one of "
                            "'darwin' (osx), 'linux', 'windows', 'raspi'")

        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_elo_rating(elo_rating)

    def select_move(self, fen):
        '''Sets stockfish state from provided fen and returns best move.
        '''
        self.stockfish.set_fen_position(fen)
        return self.stockfish.get_best_move()


class CLHumanPlayer:
    '''"Human" class that allows playing with the chessboard through CLI.
    '''
    def __init__(self):
        pass

    @staticmethod
    def select_move(board):
        '''Prompts user to select a move.
        '''
        legal_moves = board.valid_moves_from_position()
        move = None
        while move is None:
            try_move = input("Please input your move (xyxy): ").lower()
            if try_move in legal_moves:
                move = try_move
            else:
                print(f"The move {try_move} is invalid; please use format (xyxy) e.g., d2d4")
        return move


# class PhysicalHumanPlayer:
#     def __init__(self):
#         pass


# class WebHumanPlayer:
#     def __init__(self):
#         pass


# class SpeechHumanPlayer:
#     def __init__(self):
#         pass
