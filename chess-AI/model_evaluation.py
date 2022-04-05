"""Helper file for evaluating quality of our learned models.

These tests help us decide which models to keep during learning.
"""
import chess
import torch

from mcts import MCTS
from util import create_stockfish_wrapper


class Knightr0Player:
    '''"AI" class that is a wrapper around our custom modification of AlphaZero.
    '''
    def __init__(self, path_to_model):
        self.model = torch.load(path_to_model)
        # TODO: Update exploration rate.
        self.mcts = MCTS(exploration_rate=0.01)

    def select_move(self, board, fen):
        '''Sets stockfish state from provided fen and returns best move.
        '''
        return mcts.get_best_move(fen, model)


class StockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine.
    """
    def __init__(self, elo_rating=1300):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

    def select_move(self, board):
        """Sets stockfish state from provided fen and returns best move.
        """
        self.stockfish.set_fen_position(board.fen())
        return self.stockfish.get_best_move()

    def __str__(self):
        return "StockfishPlayer"


class RandomStockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine, generating random moves.
    """
    def __init__(self, elo_rating=1300):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

    def select_move(self, board):
        """Sets stockfish state from provided fen and returns random move.
        """
        self.stockfish.set_fen_position(board.fen())
        moves = self.stockfish.get_top_moves(len(list(board.legal_moves)))
        return moves[random.randint(0, len(moves) - 1)]

    def __str__(self):
        return "RandomStockfishPlayer"


def play_game(player1, player2):
    pass


def evaluate_two_players(player1, player2):
    board = chess.Board()


def main():
    # play tournament between some number of AIs (new model, previous best model, random stockfish player, weak stockfish, strong stockfish) and save pgn of each game
    # compute elo scores using BayesElo program and write ratings to a file
    # return dict of elo scores for each 

    # to stop relative elo from changing over time, we normalize elo w.r.t. 1000 rated stockfish
    # we want weak stockfish to always be rated 1000 and strong stockfish to be rated 2500




