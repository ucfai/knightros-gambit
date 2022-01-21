from functools import reduce
import random
import time

import chess
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from mcts import Mcts
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter
from state_representation import get_cnn_input


class MctsTrain:
    """This class is used to run the monte carlo simulations and drive the model training.

    Attributes:
        mcts_simulations: Number of simulations to use for MCTS
        training_examples: List of all the training examples to be used
        mcts: References the MCTS class
        policy_converter: References the PlayNetworkPolicyConverter class
    """
    def __init__(self, mcts_simulations, exploration, lr):
        # Set the number of Monte Carlo Simulations
        self.mcts_simulations = mcts_simulations
        self.mcts = Mcts(exploration)

        self.policy_converter = PlayNetworkPolicyConverter()

        # Learning rate
        self.lr = lr

    def training_game(self, nnet):
        # Stores final probability distribution from MCTS
        mcts_probs = []

        # Store fen_strings for board states
        board_fens = []

        board = chess.Board()
        while True:
            fen_string = board.fen()
            board_fens.append(fen_string)

            # Perform mcts simulations
            for _ in range(self.mcts_simulations):
                self.mcts.search(board, nnet)

            # Gets the moves and policy from the mcts, as well as the individual move to take
            # NOTE: moves[i] corresponds to search_probs[i]
            moves, search_probs, move = self.mcts.find_search_probs(fen_string, temperature=5)

            # Converts mcts search probabilites to (8,8,73) vector
            full_search_probs = self.policy_converter.compute_full_search_probs(moves, search_probs, board)
            mcts_probs.append(full_search_probs)

            # Makes the random action on the board, and gets fen string
            move = chess.Move.from_uci(move)
            board.push(move)

            # If the game is over, end the episode
            # TODO: Consider removing `board.can_claim_draw()` as it may be slow to check.
            # See https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
            if board.is_game_over() or board.can_claim_draw():
                break

        state_values = self.assign_rewards(board, len(mcts_probs))
        return board_fens, state_values, mcts_probs

    def assign_rewards(self, board, length):
        """Iterates through training examples and assigns rewards based on result of the game.
        """
        reward = 0
        values = [0 for _ in range(length)]

        if (board.outcome() is not None) and (board.outcome().winner is not None):
            reward = -1
        for move_num in range(length - 1, -1, -1):
            reward *= -1
            values[move_num] = reward

        # For demonstration print the board and outcome
        # print(board)
        print(values)
        print(board.outcome())
        return values

    def training_episode(self, nnet, games, epochs, batch_size):
        """Builds dataset from given number of training games and current network,
        then trains the network on the MCTS output and game outcomes.
        """

        with torch.no_grad():
            # Obtain data from games and separate into appropriate lists
            game_data = [self.training_game(nnet) for _ in range(games)]
            board_fens, state_values, mcts_probs = reduce(lambda g1, g2: (x+y for x, y in zip(g1, g2)), game_data)

            inputs = torch.stack([get_cnn_input(chess.Board(fen)) for fen in board_fens])
            mcts_probs = torch.tensor(np.array(mcts_probs)).float()
            state_values = torch.tensor(np.array(state_values)).float()

            # Create iterable dataset with mcts labels
            dataset = TensorDataset(inputs, state_values, mcts_probs)
            train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            # Define loss functions
            ce_loss_fn = torch.nn.CrossEntropyLoss()
            mse_loss_fn = torch.nn.MSELoss()

            # Create optimizer for updating parameters during training.
            opt = torch.optim.SGD(nnet.parameters(), lr=self.lr, weight_decay=0.001, momentum=0.9)

            # Training Loop
            for _ in range(epochs):
                losses = []
                for (inputs, state_values, mcts_probs) in train_dl:
                    with torch.enable_grad():
                        policy_batch = []
                        value_batch = []

                        # Store policies and values for entire batch
                        for state in inputs:
                            policy, value = nnet(state)
                            policy_batch.append(policy)
                            value_batch.append(value)

                        # Convert the list of tensors to a single tensor for policy and value.
                        policy_batch = torch.stack(policy_batch).float()
                        value_batch = torch.stack(value_batch).flatten().float()

                        # Find the loss and store it
                        loss = ce_loss_fn(policy_batch, mcts_probs) + mse_loss_fn(value_batch, state_values)
                        losses.append(loss.item())

                        # Calculate Gradients
                        loss.backward()

                        # Update parameters
                        opt.step()

                        # Reset gradients
                        opt.zero_grad()

                print(losses)


def main():
    # Gets the neural network, and performs episodes
    nnet = PlayNetwork()
    train = MctsTrain(mcts_simulations=3, exploration=5, lr=0.1)

    for _ in range(10):
        train.training_episode(nnet, 2, 2, 8)


if __name__ == "__main__":
    main()
 