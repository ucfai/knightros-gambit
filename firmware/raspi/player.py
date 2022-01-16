'''This file contains classes that wrap players of different types.

This class will wrap the custom AI we build, but for now we use it to get moves from stockfish.
'''
import numpy as np
import torch

# TODO: Uncomment this import when mcts implementation is finished and integrated.
# from mcts import MCTS
from util import create_stockfish_wrapper

class StockfishPlayer:
    '''"AI" class that is a simple wrapper around the Stockfish engine.
    '''
    def __init__(self, elo_rating=1300):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

    def select_move(self, fen):
        '''Sets stockfish state from provided fen and returns best move.
        '''
        self.stockfish.set_fen_position(fen)
        return self.stockfish.get_best_move()

# TODO: Uncomment this import when mcts files and model from `chess-AI` have been moved to
# firmware directory.
# class Knightr0Player:
#     '''"AI" class that is a wrapper around our custom modification of AlphaZero.
#     '''
#     def __init__(self, path_to_model):
#         # TODO: Decide where to store the model, i.e. do we upload to GitHub and have
#         # hard coded path here, or is it best to just pass from game.py. For now, just
#         # pass from game.py
#         self.model = torch.load(path_to_model)
#         # TODO: Update exploration rate.
#         self.mcts = MCTS(exploration_rate=0.01)

#     def select_move(self, board, fen):
#         '''Sets stockfish state from provided fen and returns best move.
#         '''
#         # TODO: need to implement using past n (AlphaZero uses 7?) board states as input to NN.
#         return mcts.get_best_move(fen, model)

class CLHumanPlayer:
    '''"Human" class that allows playing with the chessboard through CLI.
    '''
    def __init__(self):
        pass

    @staticmethod
    def select_move(board):
        '''Prompts user to select a move.
        '''
        uci_move = None
        while uci_move is None:
            input_move = input("Please input your move (xyxy): ").lower()
            if board.is_valid_move(input_move):
                uci_move = input_move
            else:
                print(f"The move {input_move} is invalid; please use format (xyxy) e.g., d2d4")
        return uci_move


# class PhysicalHumanPlayer:
#     def __init__(self):
#         pass


# class WebHumanPlayer:
#     def __init__(self):
#         pass


# class SpeechHumanPlayer:
#     def __init__(self):
#         pass
