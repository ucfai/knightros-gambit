"""This file holds the StockfishTrain class, used to augment our self-play training approach.
"""
import math
import random

import chess
import numpy as np
import torch

import util

class StockfishTrain:
    """Build a dataset of moves and stockfish evaluations.

    This dataset is used to decrease training time for the Knightr0 AI. After initializing the
    weights of the neural network using the Stockfish evaluations, we use self-play to
    complete training.

    Attributes:
        stockfish: the stockfish object
    """

    def __init__(self, elo=1000, depth=3):
        """Create wrapper around Stockfish engine.

        Attributes:
            elo: int specifying strength (elo) of stockfish engine
            depth: int specifying how many levels deep stockfish should search on each move
        """
        self.stockfish = util.create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_depth(depth)

    @staticmethod
    def centipawn_to_winprob(centipawn):
        """Transforms centipawn value to approximate probability of winning
        """
        return 1 / (1 + 10**(-centipawn/400))

    @staticmethod
    def choose_move(moves, epsilon):
        """Choose the next move to make with an epsilon-greedy strategy.

        To encourage exploration, a random move is chosen with probability 1-epsilon.
        Otherwise, the best action (according to the stockfish evaluation) is chosen.

        Attributes:
            moves: A list of legal moves, ordered by Stockfish evaluation from best to worst
        """

        # Return a random number in the range (0,1)
        rand = np.random.rand()

        # With probability 1-epsilon, choose the best move
        if rand < epsilon:
            move = moves[0]
        # With probability epsilon, choose a random move (to encourage exploration)
        else:
            move = moves[random.randint(0, len(moves) - 1)]

        return move

    def get_value(self, board, sig=True):
        """Return a Stockfish evaluation of the given position.

        The optional `sig` parameter specifies whether to softmax
        the return value (by default, the value is softmaxed).
        """

        # get_evaluation() always from white perspective, need to account for that
        # Positive is advantage white, negative is advantage black

        player = 1 if board.turn == chess.WHITE else -1

        # Player will transform the value depending on if black or white
        state_value = self.stockfish.get_evaluation()["value"] * player

        # Transform state value to be in the range [-1, 1]
        if sig:
            state_value = self.centipawn_to_winprob(state_value) * 2 - 1

        return state_value

    def get_move_probs(self, board, epsilon=0.3, temperature=0.1):
        """Gets the move probabilities from a position using stockfish get_top_moves
        """

        fen_string = board.fen()

        self.stockfish.set_fen_position(fen_string)

        # Gets all the moves in order from best to worst
        top_moves = self.stockfish.get_top_moves(len(list(board.legal_moves)))

        # Create list of moves
        moves = [move["Move"] for move in top_moves]

        search_probs = []
        player = 1 if board.turn == chess.WHITE else -1

        for move in top_moves:
            # if move["Centipawn"] is none, that means that there is
            # a potential mate in (x) number of moves
            # if x > 0 , that is a mate for white in x moves
            # if x < 0, that is a mate for black in x moves
            if move["Centipawn"] is None:
                search_probs.append(np.sign(move["Mate"]) * player / 2 + 0.5)
            else:
                search_probs.append(self.centipawn_to_winprob(move["Centipawn"] * player))

        search_probs = torch.tensor(search_probs).float() ** (1/temperature)

        if not torch.sum(search_probs).is_nonzero():
            search_probs = torch.full(search_probs.size(), 1/search_probs.size(dim=0))
        else:
            search_probs = search_probs / torch.sum(search_probs)

        # Will choose the move to make from the list of moves
        move = StockfishTrain.choose_move(moves, epsilon)

        return moves, search_probs, move
