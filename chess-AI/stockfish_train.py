import random
import chess
import numpy as np
import torch
from stockfish import Stockfish

from output_representation import PlayNetworkPolicyConverter


class StockfishTrain:
    """Build a dataset of moves and stockfish evaluations.

    This dataset is used to decrease training time for the Knightr0 AI. After initializing the
    weights of the neural network using the Stockfish evaluations, we use self-play to
    complete training.

    Attributes:
        stockfish: the stockfish object
        policy_converter: reference to the Policy Converter Class
    """

    def __init__(self, path):
        self.stockfish = Stockfish(path)
        self.policy_converter = PlayNetworkPolicyConverter()

    def set_params(self):
        """Sets the elo and depth for stockfish using the dashboard

        Attributes:
          dashboard: reference to the streamlit dashboard (this is temporary)
        """
        
        # Set the elo and depth of the stockfish object
        self.stockfish.set_elo_rating(1000)
        self.stockfish.set_depth(3)

    def sig(self, value, scale):
        """Calculate the sigmoid of a value

        Attributes:
            value: The value to take the sigmoid of
            scale: How much to scale the value by
        """

        return 1 / (1 + np.exp(value/scale))

    def choose_move(self, moves, epsilon):
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
            state_value = 1 - 2 * self.sig(state_value, 1)

        return state_value

    def get_move_probs(self, board, epsilon=0.3):
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
            if move["Centipawn"] == None:
                # large value * (1/move["mate"]) * player will arrange search probs
                # based on best mate
                search_probs.append(10000000 * (1 / move["Mate"]) * player)
            else:
                # TODO: CHANGE 75 TO HYPERPARAMETER
                search_probs.append(move["Centipawn"] * player / 75)

        search_probs = torch.nn.functional.softmax(torch.tensor(search_probs).float(), dim=0)
        print(search_probs)

        # Will choose the move to make from the list of moves
        move = self.choose_move(moves, epsilon)

        return moves, search_probs, move
