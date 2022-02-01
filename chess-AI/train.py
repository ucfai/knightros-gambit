from functools import reduce

import chess
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from mcts import Mcts
from ai_io import save_model, load_model
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter
from state_representation import get_cnn_input


class Train:
    """This class is used to run the monte carlo simulations and drive the model training.

    Attributes:
        mcts_simulations: Number of simulations to use for MCTS
        training_examples: List of all the training examples to be used
        mcts: References the MCTS class
        policy_converter: References the PlayNetworkPolicyConverter class
    """
    def __init__(self, lr, move_approximator, save_path, val_approximator=None):
        self.policy_converter = PlayNetworkPolicyConverter()

        # Learning rate
        self.lr = lr

        # board -> move_list, move_probabilities, move_to_take
        self.move_approximator = move_approximator

        # board -> state_value
        # if None, values will be based on the game outcome
        self.val_approximator = val_approximator

        # Specify where to save model and what model to load.
        self.save_path = save_path

    def training_game(self):
        """Run a full game, storing states, state values and policies for each state.

        Returns a tuple of 3 lists where the ith element in each list corresponds to the same board state:
            1) fen strings representing board state
            2) state value predictions
            3) policy values for all legal moves
        """

        # Stores probability distributions and values from the approximators
        all_move_probs = []
        state_values = []

        # Store fen_strings for board states
        board_fens = []

        board = chess.Board()
        while True:
            fen_string = board.fen()
            board_fens.append(fen_string)

            # Gets the moves and policy from the approximator, as well as the individual move to take
            # NOTE: moves[i] corresponds to search_probs[i]
            moves, move_probs, move = self.move_approximator(board)

            # Converts mcts search probabilites to (8,8,73) vector
            move_probs_vector = self.policy_converter.compute_full_search_probs(moves, move_probs, board)
            all_move_probs.append(move_probs_vector)

            if self.val_approximator is not None:
                state_values.append(self.val_approximator(board))

            # Makes the random action on the board, and gets fen string
            move = chess.Move.from_uci(move)
            board.push(move)

            # If the game is over, end the episode
            # TODO: Consider removing `board.can_claim_draw()` as it may be slow to check.
            # See https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
            if board.is_game_over() or board.can_claim_draw():
                break

        if self.val_approximator is None:
            state_values = self.assign_rewards(board, len(all_move_probs))

        return board_fens, state_values, all_move_probs

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

    def training_episode(self, nnet, games, epochs, batch_size, num_saved_models, overwrite_save):
        """Builds dataset from given number of training games and current network,
        then trains the network on the MCTS output and game outcomes.
        """

        with torch.no_grad():
            # Obtain data from games and separate into appropriate lists
            game_data = [self.training_game() for _ in range(games)]
            board_fens, state_values, move_probs = reduce(lambda g1, g2: (x+y for x, y in zip(g1, g2)), game_data)

            inputs = torch.stack([get_cnn_input(chess.Board(fen)) for fen in board_fens])
            move_probs = torch.tensor(np.array(move_probs)).float()
            state_values = torch.tensor(np.array(state_values)).float()

            # Create iterable dataset from game data
            dataset = TensorDataset(inputs, state_values, move_probs)
            train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            # Define loss functions
            ce_loss_fn = torch.nn.CrossEntropyLoss()
            mse_loss_fn = torch.nn.MSELoss()

            # Create optimizer for updating parameters during training.
            opt = torch.optim.SGD(nnet.parameters(), lr=self.lr, weight_decay=0.001, momentum=0.9)

            # Training Loop
            for _ in range(epochs):
                losses = []
                for (inputs, state_values, move_probs) in train_dl:
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
                        loss = ce_loss_fn(policy_batch, move_probs) + mse_loss_fn(value_batch, state_values)
                        losses.append(loss.item())

                        # Calculate Gradients
                        loss.backward()

                        # Update parameters
                        opt.step()

                        # Reset gradients
                        opt.zero_grad()
                print(losses)
            # Saves model to specified file, or a new file if not specified.
            if self.save_path != None:
                save_model(nnet, self.save_path)
            else:
                for i in range(num_saved_models):
                    if not(os.path.isfile(f'chess-AI/models-{i+1}.pt')):
                        if overwrite_save and i != 0:
                            save_model(nnet, f'chess-AI/models-{i}.pt')
                            break
                        save_model(nnet, f'chess-AI/models-{i+1}.pt')
                        break
                    if i == num_saved_models - 1:
                        save_model(nnet, f'chess-AI/models-{num_saved_models}.pt')

def main():
    mcts_simulations = 3
    num_saved_models = 5
    load_path = None
    overwrite_save = True
    mcts = Mcts(exploration=5)
    stockfish = StockfishTrain()
    nnet = PlayNetwork()
    nnet.train()

    if load_path != None:
        nnet = load_model(nnet, load_path)
    else:
        for i in range(num_saved_models):
            if not(os.path.isfile(f'chess-AI/models-{i+2}.pt')):
                if i != 0:
                    nnet = load_model(nnet, f'chess-AI/models-{i+1}.pt')
                break
        

    # Partially applies parameters to mcts function
    
    for _ in range(stocktrain_amt):
        mcts_moves = lambda board: mcts.get_tree_results(mcts_simulations, nnet, board, temperature=5)
        train = Train(lr=0.2, move_approximator=mcts_moves, save_path=None)
        train.training_episode(nnet, games=3, epochs=3, batch_size=10, num_saved_models=num_saved_models, overwrite_save=overwrite_save)

    
    for _ in range(mcts_amt):
        value_approximator = stockfish.value_approximator()
        mcts_moves = lambda board: stockfish.get_move_probs(board)
        train = Train(lr=0.2, move_approximator=mcts_moves,val_approximator=value_approximator, save_path=None)
        train.training_episode(nnet, games=3, epochs=3, batch_size=10, num_saved_models=num_saved_models, overwrite_save=overwrite_save)


if __name__ == "__main__":
    main()
 