import random
import numpy as np
from stockfish import Stockfish
import chess
import torch

from output_representation import PlayNetworkPolicyConverter


class StockfishTrain:
    """Class with all the necessary functionialities for
    building a dataset using stockfish

    Attributes:
        stockfish: the stockfish object
        policy_converter: reference to the Policy Converter Class
    """

    def __init__(self, path):
        self.stockfish = Stockfish(path)
        self.policy_converter = PlayNetworkPolicyConverter()

    def set_params(self):
        """ Sets the elo and depth for stockfish using the dashboard

        Parameters:
        dashboard: reference to the streamlit dashboard (this is temporary)
        """
        
        # Set the elo and depth of the stockfish object
        self.stockfish.set_elo_rating(1000)
        self.stockfish.set_depth(3)

    def sig(self, value, scale):
        """Calculate the sigmoid of a value

        Parameters:
        value: The value to take the sigmoid of
        scale: How much to scale the value by
        """

        return 1 / (1 + np.exp(value/scale))

    def choose_move(self, moves):
        """ Chooses the next move to make
        We want to make sure the top move isn't always what is
        chosen

        Parameters:
        moves: The list of posssible moves to take
        """

        epsilon = 0.3
        
        # Return a random number in the range (0,1)
        rand = np.random.rand()

        # Choose the best move 30% of the time
        if rand < epsilon:
            move = moves[0]
        # Choose a random move 70% of the time , allows for exploration
        else:
            move = moves[random.randint(0, len(moves) - 1)]

        return move

    def get_value(self, board, sig=True):
        """ Returns a value for a given state on the board
        Utilizes stockfish Centipawn calculation through stockfish.get_evaluation()
        Will then use sig() to transform the value between (0,1)
        """

        # get_evaluation() always from white perspective, need to account for that
        # Positive is advantage white, negative is advantage black

        player = 1 if board.turn == chess.WHITE else -1
        
        # Player will transform the value depending on if black or white
        state_value = self.stockfish.get_evaluation()["value"] * player

        # Will use the sig() and transform between (0,1)
        if sig == True:
            state_value = 1 - 2 * self.sig(state_value, 1)

        return state_value

    def get_move_probs(self, board):
        """Gets the move probabilities from a position using stockfish get_top_moves
        """

        fen_string = board.fen()

        self.stockfish.set_fen_position(fen_string)

        # Gets all the moves in order from best to worst
        top_moves = self.stockfish.get_top_moves(len(list(board.legal_moves)))

        # Create list of moves
        moves = [move["Move"] for move in top_moves]

        # Best move should have a prob of 1, where all others should be 0
        # NOTE: Maybe use some sort of prob distribution
        search_probs = np.empty_like(top_moves)
        for move in range(top_moves):
            search_probs[move] = self.get_value(chess.Board(fen_string).push(chess.Move.from_uci(top_moves[move])), sig=False)

        search_probs = torch.nn.Softmax(torch.from_numpy(search_probs))
        print("Search probs:")
        print(search_probs)
        print()
        
        # search_probs = [0 for _ in top_moves]
        # search_probs[0] = 1

        # Will choose the move to make from the list of moves
        move = self.choose_move(moves)

        return moves, search_probs, move
