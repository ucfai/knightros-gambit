import random
import numpy as np
from stockfish import Stockfish
import chess

from output_representation import PlayNetworkPolicyConverter


class StockfishTrain:
    """Class with all the necessary functions for
    building a dataset using stockfish

    Attributes:
        stockfish: The stockfish object
        policy_converter: Reference to the Policy Converter Class
    """

    def __init__(self, path):
        self.stockfish = Stockfish(path)
        self.policy_converter = PlayNetworkPolicyConverter()

    def set_params(self, dashboard):
        """ Sets the elo and depth for stockfish using the dashboard
        """

        elo, depth = dashboard.configure_stockfish()
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_depth(depth)

    def sig(self, value, scale):
        """Calculate the sigmoid of a value

        Parameters:
        value: The value to take the sigmoid of
        scale: How much to scale the value by
        """

        return 1 / (1 + np.exp(value/scale))

    def choose_move(self, moves):
        """Chooses a move to make
        Ensures that the top move isn't always the one that is chosen

        Parameters:
        moves: The list of posisble moves to take
        """

        epsilon = 0.3
        rand = np.random.rand()

        if rand < epsilon:
            move = moves[0]
        else:
            move = moves[random.randint(0, len(moves) - 1)]

        return move

    def get_value(self, board):
        """ Returns a value for a given state on the board
        Utilizes stockfish Centipawn calculation through stockfish.get_evaluation()
        """

        # get_evaluation() alway from white perspective, need to account for that
        player = 1 if board.turn == chess.WHITE else -1

        value = self.stockfish.get_evaluation()["value"] * player
        value = 1 - 2 * self.sig(value, 1)

        return value

    def get_move_probs(self, board):
        """Gets the move probabilities from a position using stockfish get_top_moves
        """

        fen_string = board.fen()
        self.stockfish.set_fen_position(fen_string)

        # Needs to get all the moves in order from best to worst
        top_moves = self.stockfish.get_top_moves(len(list(board.legal_moves)))

        # Get list of moves
        moves = [move["Move"] for move in top_moves]

        # Best move should have a prob of 1, where all others should be 0
        # NOTE: Maybe use some sort of prob distribution
        search_probs = [0 for _ in top_moves]
        search_probs[0] = 1

        # Will choose the move to make from the list of moves
        move = self.choose_move(moves)

        return moves, search_probs, move
