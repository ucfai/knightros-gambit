import random

import chess
import torch
import numpy as np
import time
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

        # Stores mcts probabilities and outcome reward values
        self.mcts_probs = []
        self.mcts_evals = []

        # Stores board fen strings
        self.board_fens = []

        self.mcts = Mcts(exploration)
        self.policy_converter = PlayNetworkPolicyConverter()

        # Learning rate
        self.lr = lr

    def training_episode(self, nnet, epochs):
        with torch.no_grad():
            # At each episode need to set the training example and network predictions to empty
            self.mcts_probs = []
            self.mcts_evals = []
            self.board_fens = []

            batch_size = 8
            num = 0

            for _ in range(epochs):
                # The board needs to be initialized to the starting state
                board = chess.Board()

                while True:
                    fen_string = board.fen()

                    # Perform mcts simulations
                    for _ in range(self.mcts_simulations):
                        self.mcts.search(board, nnet)

                    # Gets the moves and policy from the mcts, as well as the individual move to take
                    # NOTE: moves[i] corresponds to search_probs[i]
                    moves, search_probs, move = self.mcts.find_search_probs(fen_string, temperature=5)
            
                    # Store board state (used in training loop)
                    self.board_fens.append(fen_string)

                    # Converts mcts search probabilites to (8,8,73) vector
                    full_search_probs = self.policy_converter.compute_full_search_probs(moves, search_probs, board)
                    
                    # Adds entry to the training examples
                    self.mcts_probs.append(full_search_probs)

                    # Makes the random action on the board, and gets fen string
                    move = chess.Move.from_uci(move)
                    board.push(move)
                    # print(board)

                    # If the game is over, end the episode
                    if board.is_game_over() or board.is_stalemate() or board.is_seventyfive_moves() or board.is_fivefold_repetition() or board.can_claim_draw():
                        self.mcts_evals = torch.zeros(len(self.mcts_probs))
                        self.assign_rewards(board)
                        break

                self.mcts_probs = torch.tensor(np.array(self.mcts_probs)).float()
                self.mcts_evals = self.mcts_evals.float()

                # Create iterable dataset with mcts labels
                dataset = TensorDataset(self.mcts_probs, self.mcts_evals)
                train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

                # Define loss functions
                ce_loss_fn = torch.nn.CrossEntropyLoss()
                mse_loss_fn = torch.nn.MSELoss()

                # Create optimizer for updating parameters during training.
                opt = torch.optim.SGD(nnet.parameters(), lr=self.lr, weight_decay=0.001, momentum=0.9)

                # Training Loop
                for _ in range(epochs):
                    losses = []
                    for batch_index, (mcts_probs, mcts_evals) in enumerate(train_dl):
                        with torch.enable_grad():
                            # Get the batch size based on the batch size of the training dataloader
                            batch_size = mcts_evals.size(dim=0)

                            policy_batch = []
                            value_batch = []
                            
                            # Store policies and values for entire batch
                            for i in range(batch_size):
                                policy, value = nnet(get_cnn_input(chess.Board(self.board_fens[i + batch_index * 8])))
                                policy_batch.append(policy)
                                value_batch.append(value)

                            # Convert the list of tensors to a single tensor for policy and value.
                            policy_batch = torch.stack(policy_batch).float()
                            value_batch = torch.stack(value_batch).flatten().float()

                            # Find the loss and store it
                            loss = ce_loss_fn(policy_batch, mcts_probs) + mse_loss_fn(value_batch, mcts_evals)
                            losses.append(loss.item())

                            # Calculate Gradients
                            loss.backward()

                            # Update parameters
                            opt.step()

                            # Reset gradients
                            opt.zero_grad()

                    print(losses)
            
    def assign_rewards(self, board):
        """Iterates through training examples and assigns rewards based on result of the game.
        """
        reward = 0
        if (board.outcome() is not None) and (board.outcome().winner is not None):
            reward = -1
        for move_num in range(len(self.mcts_evals) - 1, -1, -1):
            reward *= -1
            self.mcts_evals[move_num] = reward

        # For demonstration print the board and outcome
        # print(board)
        print(self.mcts_evals)
        print(board.outcome())


def main():
    # Gets the neural network, and performs and episode
    nnet = PlayNetwork()
    train = MctsTrain(mcts_simulations=5, exploration=5, lr=0.1)

    # TODO: Make training support epochs
    for _ in range(10):
        train.training_episode(nnet, 1)


if __name__ == "__main__":
    main()
 