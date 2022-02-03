from functools import reduce

import chess
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from zmq import device

from mcts import Mcts
from ai_io import save_model, load_model
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter
from state_representation import get_cnn_input
from stockfish_train import StockfishTrain
from streamlit_dashboard import StreamlitDashboard


class Train:
    """This class is used to run the monte carlo simulations and drive the model training.

    Attributes:
        mcts_simulations: Number of simulations to use for MCTS
        training_examples: List of all the training examples to be used
        mcts: References the MCTS class
        policy_converter: References the PlayNetworkPolicyConverter class
    """
    def __init__(self, lr, move_approximator, save_path, device, val_approximator=None):
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

        self.device = device

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

        return values

    def create_dataset(self,games):
        """Builds dataset from given number of training games and current network,
        then trains the network on the MCTS output and game outcomes.
        """

        with torch.no_grad():
            # Obtain data from games and separate into appropriate lists
            game_data = [self.training_game() for _ in range(games)]

            inputs = torch.stack([get_cnn_input(chess.Board(game[0])) for game in game_data])
            state_values = torch.tensor([game[1] for game in game_data])
            move_probs = torch.tensor([game[2] for game in game_data])

            # Create iterable dataset from game data
            dataset = TensorDataset(inputs, state_values, move_probs)

        return dataset
            # Builds the dataset

    def trainon_dataset(self,dataset,dashboard, nnet, epochs, batch_size, num_saved_models, overwrite_save):
        with torch.no_grad():

            # Holds the average losses for graphing purposes
            average_pol_loss = []
            average_val_loss = []

            # Define loss functions
            ce_loss_fn = torch.nn.CrossEntropyLoss()
            mse_loss_fn = torch.nn.MSELoss()

            # Create optimizer for updating parameters during training.
            opt = torch.optim.SGD(nnet.parameters(), lr=self.lr, weight_decay=0.001, momentum=0.9)
            train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            # Training Loop
            for e in range(epochs):
                start = time.time()
                value_losses = []
                policy_losses = []
                losses = []
                num_moves = 0
                for (inputs, state_values, move_probs) in train_dl:
                    num_moves += 1

                    with torch.enable_grad():
                        policy_batch = []
                        value_batch = []

                        # Store policies and values for entire batch
                        for state in inputs:
                            policy, value = nnet(state.to(device=self.device))
                            policy_batch.append(policy)
                            value_batch.append(value)

                        # Convert the list of tensors to a single tensor for policy and value.
                        policy_batch = torch.stack(policy_batch).float().to(self.device)
                        value_batch = torch.stack(value_batch).flatten().float().to(self.device)

                        move_probs = move_probs.to(device=self.device)
                        state_values = state_values.to(device=self.device)

                        # Find the loss and store it

                        pol_loss = ce_loss_fn(policy_batch, move_probs)
                        val_loss = mse_loss_fn(value_batch, state_values)
                        policy_losses.append(pol_loss)
                        value_losses.append(val_loss)

                        loss = pol_loss + val_loss

                        losses.append(loss.item())

                        # Calculate Gradients
                        loss.backward()

                        # Update parameters
                        opt.step()

                        # Reset gradients
                        opt.zero_grad()

                end = time.time()

                # Calculate average losses and add it to the list
                policy_loss = sum(policy_losses)/len(policy_losses)
                value_loss =  sum(value_losses)/len(value_losses)
                average_pol_loss.append(policy_loss.cpu())
                average_val_loss.append(value_loss.cpu())

            dashboard.visualize_epochs(policy_loss,value_loss,end,start,num_moves,e)

        dashboard.visualize_training_stats(average_pol_loss,average_val_loss)

        # Saves model to specified file, or a new file if not specified.
        if self.save_path != None:
            save_model(nnet, self.save_path)
        else:
            for i in range(num_saved_models):
                if not(os.path.isfile(f'models/models-{i+1}.pt')):
                    if overwrite_save and i != 0:
                        save_model(nnet, f'models/models-{i}.pt')
                        break
                    save_model(nnet, f'models/models-{i+1}.pt')
                    break
                if i == num_saved_models - 1:
                    save_model(nnet, f'models/models-{num_saved_models}.pt')
def main():

    # Detect device to train on
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    dashboard = StreamlitDashboard()

    stockfish_path = dashboard.set_stockfish_path()
    stockfish = StockfishTrain(stockfish_path)
    stockfish.set_params(dashboard)

    stocktrain_games,stocktrain_epochs,mcts_games,mcts_epochs = dashboard.set_training_data()

    #TODO: Figure out how many times to perform self play
    mcts_amt = 5
    mcts_simulations,exploration = dashboard.set_mcts_params()

    mcts = Mcts(exploration)

    batch_size,lr = dashboard.set_nnet_hyperparamters()
    model_path = dashboard.load_path()
    dataset_path = dashboard.load_dataset()

    num_saved_models = 5
    overwrite_save = True

    nnet = PlayNetwork().to(device=device)

    if model_path != None:
        nnet = load_model(nnet, model_path)
    else:
        for i in range(num_saved_models):
            if not(os.path.isfile(f'chess-AI/models-{i+2}.pt')):
                if i != 0:
                    nnet = load_model(nnet, f'chess-AI/models-{i+1}.pt')
                break     

    if(dashboard.train_button()):

        value_approximator = lambda board: stockfish.get_value(board)
        stocktrain_moves = lambda board: stockfish.get_move_probs(board)
        train = Train(lr=lr, move_approximator=stocktrain_moves, save_path=None, device=device, val_approximator=value_approximator)

        # Dataset needs to be either created or loaded
        if dataset_path:
            dataset = torch.load(dataset_path)
        else:
            dataset = train.create_dataset(stocktrain_games)
            torch.save(dataset,'datasets/stockfish_data.pt')

        # Train using the stockfish datset
        train.trainon_dataset(dataset,dashboard,nnet,epochs=stocktrain_epochs, batch_size=batch_size, num_saved_models=num_saved_models, overwrite_save=overwrite_save)

        for _ in range(mcts_amt):
            mcts_moves = lambda board: mcts.get_tree_results(mcts_simulations, nnet, board, temperature=5)
            train = Train(lr=lr, move_approximator=mcts_moves, save_path=None, device=device, val_approximator=None)
            dataset = train.create_dataset(mcts_games)
            train.trainon_dataset(dataset,dashboard,nnet, epochs=mcts_epochs, batch_size=batch_size, num_saved_models=num_saved_models, overwrite_save=overwrite_save)

    
if __name__ == "__main__":
    main()
 