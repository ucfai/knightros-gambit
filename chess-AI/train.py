import random

import chess

from mcts import Mcts

from nn_layout import PlayNetwork

from output_representation import PlayNetworkPolicyConverter

class MctsTrain: 

    """
    This class is used to run the monte carlo
    simulations and drive the model training

    Attributes:
        mcts_simulations: Number of simulations to use for MCTS
        training_examples: List of all the training examples to be used
        mcts: References the MCTS class
        policy_converter: References the PlayNetworkPolicyConverter class
    """

    def __init__(self, mcts_simulations,exploration):

        # Set the number of Monte Carlo Simulations
        self.mcts_simulations = mcts_simulations

        self.training_examples = []
        self.mcts = Mcts(exploration)
        self.policy_converter = PlayNetworkPolicyConverter()

    def training_episode(self, nnet):

        # At each episode need to set the training example to empty
        self.training_examples = []

        # The board needs to be initialized ot the starting state
        board = chess.Board()
        fen_string = board.fen()

        while True:
            # Perform mcts simulation
            for _ in range(self.mcts_simulations):
                self.mcts.search(board, nnet)

            # Need to get the moves and policy from the mcts
            # NOTE: moves[i] corresponds to search_probs[i]
            moves,search_probs = self.mcts.find_search_probs(fen_string)

            # Gets a random index from the search_probs and makes random move
            rand_move_idx = random.randint(0,len(search_probs) - 1)
            move = moves[rand_move_idx]

            # Converts mcts search probabilites to (8,8,73) vector
            converted_search_probs = self.policy_converter.compute_full_search_probs(move, search_probs, board)

            # Adds entry to the training examples
            self.training_examples.append([fen_string,converted_search_probs,None])

            # Makes the random action on the board, and gets fen string
            move  = chess.Move.from_uci(move)
            board.push(move)
            fen_string = board.fen()

            # if the game is over then that is the end of the episode
            if board.is_game_over():
                # Need to assign the rewards to the examples
                self.assign_rewards(board)
                return

    def assign_rewards(self, board): 

        """
        Iterates through training examples and 
        assigns rewards based on result of the game
        """

        for example in self.training_examples:
            if board.outcome().winner:
                example[2] = -1
            else:
                example[2] = 0

        # For demonstration print the board and outcome
        print(board)
        print(board.outcome().termination)


def main():

    # Gets the neural network, and performs and episode
    nnet = PlayNetwork()
    train = MctsTrain(2)
    train.training_episode(nnet)

if __name__ == "__main__":
    main()
