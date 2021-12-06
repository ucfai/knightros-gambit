import random
import chess
from mcts import Mcts
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter

class MctsTrain: # algorithm for training the model.

    def __init__(self,mcts_simulations):

        # Set the starting fen_string
        self.start_fen = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP"
        "/RNBQKBNR w KQkq - 0 1")

        # Set the number of Monte Carlo Simulations
        self.mcts_simulations = mcts_simulations
        self.training_examples = []

    def training_episode(self,nnet):

        # At each episode need to set the training example to empty
        self.training_examples = []

        # The board needs to be initialized ot the starting state
        board = chess.Board(self.start_fen)
        fen_string = self.start_fen
        mcts = Mcts() # assign MCTS object
        policy_converter = PlayNetworkPolicyConverter()

        while True:

            # Perform mcts simulation
            for _ in range(self.mcts_simulations):
                mcts.search(board,nnet)

            # Need to get the moves and policy from the mcts
            keys,policy = mcts.find_search_probs(fen_string)

            # Makes a random choice from the policy
            choice = random.randint(0,len(policy) - 1)
            action = keys[choice]

            # Converts mcts search probabilites to (8,8,73) vector
            policy = policy_converter.compute_full_search_probs(keys, policy, board)

            # Adds entry to the training examples
            self.training_examples.append([fen_string,policy,None])

            # Makes the random action on the board, and gets fen string
            action = chess.Move.from_uci(action)
            board.push(action)
            fen_string = board.fen()

            # if the game is over then that is the end of the episode
            if board.is_game_over():
                # Need to assign the rewards to the examples
                self.assign_rewards(board)
                return

    def assign_rewards(self,board): # assigns the reward to training examples

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
