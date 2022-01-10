import random

import chess
import torch
import numpy as np
from torch.autograd import backward
from torch.utils.data import DataLoader

from mcts import Mcts
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter
from state_representation import get_cnn_input
from self_play_dataset import SelfPlayDataset

class MctsTrain:
    """This class is used to run the monte carlo simulations and drive the model training.

    Attributes:
        mcts_simulations: Number of simulations to use for MCTS
        training_examples: List of all the training examples to be used
        mcts: References the MCTS class
        policy_converter: References the PlayNetworkPolicyConverter class
    """
    def __init__(self, mcts_simulations, exploration):
        # Set the number of Monte Carlo Simulations
        self.mcts_simulations = mcts_simulations

        # Stores mcts probabilities and outcome reward values
        self.mcts_probs = []
        self.mcts_evals = []

        # Stores network policy and evaluation predictions
        # self.nn_preds = torch.tensor([])
        self.nn_probs = []
        self.nn_evals = []

        self.mcts = Mcts(exploration)
        self.policy_converter = PlayNetworkPolicyConverter()

        # Learning rate
        self.lr = 0.2

    def training_episode(self, nnet, epochs):
        with torch.no_grad():
            # At each episode need to set the training example and network predictions to empty
            self.mcts_probs = []
            self.mcts_evals = []
            # self.nn_preds = torch.tensor([])
            self.nn_probs = []
            self.nn_evals = []

            batch_size = 1
            num = 0

            for _ in range(epochs):
                # The board needs to be initialized to the starting state
                num += 1
                num2 = 0
                print(num)
                board = chess.Board()
                fen_string = board.fen()
                # for __ in range(batch_size):
                while True:
                    num2 += 1
                    print(num2)
                    # Perform mcts simulation
                    for __ in range(self.mcts_simulations):
                        self.mcts.search(board, nnet)

                    # Need to get the moves and policy from the mcts
                    # NOTE: moves[i] corresponds to search_probs[i]
                    moves, search_probs = self.mcts.find_search_probs(fen_string)

                    # Get network predictions and store them.
                    policy, value = nnet(get_cnn_input(board).float())
                    self.nn_probs.append(policy)
                    self.nn_evals.append(value)
                    move_values = nnet.predict(policy, board)

                    # Gets a random index from the move values from the network policy and makes random move
                    # TODO: Choose move according to network move values and exploration.
                    rand_move_idx = random.randint(0, len(move_values) - 1)
                    move = moves[rand_move_idx]

                    # Converts mcts search probabilites to (8,8,73) vector
                    full_search_probs = self.policy_converter.compute_full_search_probs(moves,
                                                                                        search_probs,
                                                                                        board)
                    
                    # Adds entry to the training examples
                    self.mcts_probs.append(full_search_probs)
                    self.mcts_evals.append(None)

                    # Makes the random action on the board, and gets fen string
                    move = chess.Move.from_uci(move)
                    board.push(move)
                    fen_string = board.fen()

                    # if the game is over then that is the end of the episode
                    if board.is_game_over() or board.is_stalemate() or board.is_seventyfive_moves() or board.is_fivefold_repetition() or board.can_claim_draw():
                        # Need to assign the rewards to the examples
                        self.assign_rewards(board)
                        break
                # batch_size = len(self.training_examples)
                self.mcts_probs = torch.tensor(self.mcts_probs)
                self.mcts_evals = torch.tensor(self.mcts_evals)
                # self.nn_evals = torch.tensor(self.nn_evals)
                # self.nn_probs = torch.tensor(self.nn_probs)
                dataset = SelfPlayDataset(self.mcts_probs, self.mcts_evals, self.nn_probs, self.nn_evals)
                print(dataset[0])
                print(dataset[0][1])
                print(dataset[0][2])
                print(dataset[0][3])
                # print(dataset[1])
                train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
                

                loss_fn1 = torch.nn.CrossEntropyLoss()
                loss_fn2 = torch.nn.MSELoss()

                opt = torch.optim.SGD(nnet.parameters(), lr=self.lr)

            # TODO: change dataset to include labels.
            for _ in range(epochs):
                for batch_index, (mcts_probs, mcts_evals, nn_probs, nn_evals) in enumerate(train_dl):
                    loss = loss_fn1(nn_probs, mcts_probs) + loss_fn2(nn_evals, mcts_evals)
                    print(loss)

                    loss.backward()

                    #Update parameters
                    opt.step()

                    #Reset gradients
                    opt.zero_grad()
            
        

    def assign_rewards(self, board):
        """Iterates through training examples and assigns rewards based on result of the game.
        """
        reward = 0
        winner = True
        if board.outcome() == None:
            winner = False
        if winner and board.outcome().winner:
            reward = -1
        for move_num in range(len(self.mcts_evals) - 1, -1, -1):
            reward *= -1
            self.mcts_evals[move_num] = reward * -1

        # For demonstration print the board and outcome
        print(board)


def main():
    # Gets the neural network, and performs and episode
    nnet = PlayNetwork().cuda()
    train = MctsTrain(mcts_simulations=2, exploration=1)
    train.training_episode(nnet, 1)

if __name__ == "__main__":
    main()
