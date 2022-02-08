"""
Main training program
"""
import time
import os
import chess
import torch

from torch.utils.data import DataLoader, TensorDataset

from mcts import Mcts
from ai_io import save_model, load_model
from nn_layout import PlayNetwork
from output_representation import PlayNetworkPolicyConverter
from state_representation import get_cnn_input
from stockfish_train import StockfishTrain
from streamlit_dashboard import StreamlitDashboard

class Train:
    """The main training class, will train using stockfish and mcts
    
    1. Will create a dataset or load a dataset of (fen_strings,move_probs,values)
    generated from Stockfish
    2. Will then train using the model
    3. Then will generate training examples by self play using the MCTS
    4. Will train the model on those examples

    Attributes:
        policy_converter: References the PlayNetworkPolicyConverter class
        learning_rate: The learning rate for the model
        move_approximator: The lambda function that gets move probs
        value_approximator: The lambda function that gets the value
        save_path: The location where to save the model
        device: The device used for training (either CPU or CUDA)
    """

    def __init__(self, learning_rate, move_approximator, save_path, device, val_approximator=None):

        # Policy Converter is convert output of the model
        self.policy_converter = PlayNetworkPolicyConverter()

        # Learning rate for gradient descent while training
        self.learning_rate = learning_rate

        # board -> move_list, move_probabilities, move_to_take
        self.move_approximator = move_approximator

        # board -> state_value
        # if None, values will be based on the game outcome
        self.val_approximator = val_approximator

        # Specify path to save the model
        self.save_path = save_path

        # Specify either CPU or GPU
        self.device = device

    def training_game(self):
        """Run a full game, storing fen_strings, policies, and values

        Returns a tuple of 3 lists where the ith element in each list corresponds to the board state:
            1) fen strings representing board state
            2) policy values for all legal moves
            3) state value predictions 
        """

        # Stores probability distributions and values from the approximators
        all_move_probs = []
        all_state_values = []

        # Store fen_strings for board states
        board_fens = []

        board = chess.Board()

        while True:
            # Get current fen string and append to list
            fen_string = board.fen()
            board_fens.append(fen_string)

            # Gets the moves and policy from the approximator and move to make
            # NOTE: moves[i] corresponds to search_probs[i]
            moves, move_probs, move = self.move_approximator(board)

            # Converts move probs into a (8,8,73) vector to be used for training
            move_probs_vector = self.policy_converter.compute_full_search_probs(moves, move_probs, board)
            all_move_probs.append(move_probs_vector)

            # val_approximator will not be none for Stockfish
            if self.val_approximator is not None:
                all_state_values.append(self.val_approximator(board))

            # Makes the move on the pychess board
            move = chess.Move.from_uci(move)
            board.push(move)

            # If the game is over, end the episode
            # TODO: Consider removing `board.can_claim_draw()` as it may be slow to check.
            # See https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
            if board.is_game_over() or board.can_claim_draw():
                break

        # valu_approximator will be none for mcts
        if self.val_approximator is None:
            all_state_values = self.assign_rewards(board, len(all_move_probs))

        return board_fens, all_state_values, all_move_probs

    def assign_rewards(self, board, length):
        """Iterates through training examples and assigns rewards based on result of the game.
        NOTE: Want to triple check this logic, reward assignment is very important
        """
        reward = 0
        
        state_values = [0 for _ in range(length)]

        # If board.outcome() is none , then there is a draw
        if (board.outcome() is not None) and (board.outcome().winner is not None):
            reward = -1
        for move_num in range(length - 1, -1, -1):
            reward *= -1
            state_values[move_num] = reward

        return state_values

    def create_dataset(self, games):
        """Builds a dataset with the size of (games)

        Parameters:
        games: The total number of games to generate for the dataset

        """

        # Using tensors but do not want to keep track of gradients
        with torch.no_grad():
            # Obtain data from training games
            game_data = [self.training_game() for _ in range(games)]

        # Convert all the fen strings into tensors that are used in the dataset
        input_state = torch.stack([get_cnn_input(chess.Board(state)) for game in game_data for state in game[0]])
        state_values = torch.tensor([state_val for game in game_data for state_val in game[1]]).float()
        move_probs = torch.tensor([move_prob for game in game_data for move_prob in game[2]]).float()

        # Create iterable dataset from game data
        dataset = TensorDataset(input_state, state_values, move_probs)

        # Return the dataset to be used
        return dataset

    def train_on_dataset(self, dataset, dashboard, nnet, epochs, batch_size, num_saved_models, overwrite_save):
        """Function to train the with a dataset passed as a parameter

        Parameters:
        dataset: the dataset to use for training
        dashboard: the streamlit dashboard (this will eventually be removed)
        nnet: the neural network
        epochs: number of epochs
        batch_size: the batch size for SGD
        """

        # Stores the average losses which are used for graphing
        average_pol_loss = []
        average_val_loss = []

        # Define loss functions (cross entropy and mean squared error)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        mse_loss_fn = torch.nn.MSELoss()

        # Create optimizer for updating parameters during training
        # NOTE: We are using SGD right now, maybe consider other optimizers such as Adam
        opt = torch.optim.SGD(nnet.parameters(), lr=self.learning_rate, weight_decay=0.001, momentum=0.9)
        train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        # Main training loop
        for epoch in range(epochs):

            # Variables used solely for monitoring training, not used for actually updating the model
            start = time.time()
            value_losses = []
            policy_losses = []
            losses = []
            num_moves = 0

            # Iterate through train_dl
            # train_dl is segmented into batches of (input_states, all_state_values, move_probs)
            for (input_states, state_values, move_probs) in train_dl:
                num_moves += 1

                policy_batch = []
                value_batch = []

                # Store policies and values for entire batch
                # TODO: Loop might be replaceable with one nnet call on inputs followed by zip to separate lists
                for state in input_states:
                    policy, value = nnet(state.to(device=self.device))
                    policy_batch.append(policy)
                    value_batch.append(value)

                # Convert the list of tensors to a single tensor for policy and value.
                policy_batch = torch.stack(policy_batch).float().to(self.device)
                value_batch = torch.stack(value_batch).flatten().float().to(self.device)

                move_probs = move_probs.to(device=self.device)
                state_values = state_values.to(device=self.device)

                # Compute policy loss and value loss using loss functions
                pol_loss = ce_loss_fn(policy_batch, move_probs)
                val_loss = mse_loss_fn(value_batch, state_values)
                loss = pol_loss + val_loss

                # Add to list for graphing purposes
                policy_losses.append(pol_loss)
                value_losses.append(val_loss)
                losses.append(loss.item())

                # Calculate gradients, update parameters, and reset gradients
                loss.backward()
                opt.step()
                opt.zero_grad()

            end = time.time()

            # Calculate and store the average losses
            policy_loss = sum(policy_losses) / len(policy_losses)
            value_loss = sum(value_losses) / len(value_losses)
            average_pol_loss.append(policy_loss.cpu().detach().numpy())
            average_val_loss.append(value_loss.cpu().detach().numpy())

            dashboard.visualize_epochs(policy_loss, value_loss, end, start, num_moves, epoch)

        dashboard.visualize_training_stats(average_pol_loss, average_val_loss)

        # Saves model to specified file, or a new file if not specified.
        # TODO: Figure frequency of model saving, right now it is after every epoch
        # TODO: Determine better names for the models when saving 
        if self.save_path is not None:
            save_model(nnet, self.save_path)
        else:
            # Iterate through the number of models saved
            for i in range(num_saved_models):
                if not (os.path.isfile(f'models/models-{i + 1}.pt')):
                    if overwrite_save and i != 0:
                        save_model(nnet, f'models/models-{i}.pt')
                        break
                    save_model(nnet, f'models/models-{i + 1}.pt')
                    break
                if i == num_saved_models - 1:
                    save_model(nnet, f'models/models-{num_saved_models}.pt')

def main():
    """Main function that will be run when starting training
    """

    # Detect device to train on
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # Initializes the streamlit dashboard
    # NOTE: In official training we will not be using the dashboard
    dashboard = StreamlitDashboard()


    # Get dataset parameters from dashboard
    stocktrain_games, stocktrain_epochs, mcts_games, mcts_epochs = dashboard.set_training_data()
    batch_size, learning_rate = dashboard.set_nnet_hyperparamters()

    # Gets stockfish training object, and sets parameters (elo,depth)
    stockfish_path = dashboard.set_stockfish_path()
    stockfish = StockfishTrain(stockfish_path)
    stockfish.set_params(dashboard)

    # Get MCTS object and parameters
    # TODO: Figure out how many times to perform self play (and better name for this variable)
    mcts_amt = 5
    mcts_amt, mcts_simulations, exploration = dashboard.set_mcts_params()
    mcts = Mcts(exploration)

    nnet = PlayNetwork().to(device=device)

    # TODO: Allow user choice for these values
    num_saved_models = 5
    overwrite_save = True

    # Will get the paths to load models and datasets from
    model_path = dashboard.load_path()
    dataset_path = dashboard.load_dataset()

    # Load in a model
    if model_path is not None:
        nnet = load_model(nnet, model_path)
    else:
        for i in range(num_saved_models):
            if not (os.path.isfile(f'chess-AI/models-{i + 2}.pt')):
                if i != 0:
                    nnet = load_model(nnet, f'chess-AI/models-{i + 1}.pt')
                break

    if dashboard.train_button():

        # Value and move approximators from stockfish
        value_approximator = stockfish.get_value
        stocktrain_moves = stockfish.get_move_probs

        # Training object instantiated using stockfish data
        train = Train(learning_rate=learning_rate, move_approximator=stocktrain_moves, save_path=None, device=device,
                      val_approximator=value_approximator)

        # Dataset needs to be either created or loaded
        if dataset_path:
            dataset = torch.load(dataset_path)
        else:
            dataset = train.create_dataset(stocktrain_games)
            # NOTE: Dataset should be given a more descriptive name, this is just temporary
            torch.save(dataset, 'datasets/stockfish_data.pt')

        # Train using the stockfish dataset
        train.train_on_dataset(dataset, dashboard, nnet, epochs=stocktrain_epochs,batch_size=batch_size,
                               num_saved_models=num_saved_models, overwrite_save=overwrite_save)

        # Will iterate through the number of training episodes
        for _ in range(mcts_amt):
            mcts_moves = lambda board: mcts.get_tree_results(mcts_simulations, nnet, board, temperature=5)
            train = Train(learning_rate=learning_rate, move_approximator=mcts_moves, save_path=None, device=device,
                          val_approximator=None)
                          
            dataset = train.create_dataset(mcts_games)
            train.train_on_dataset(dataset, dashboard, nnet, epochs=mcts_epochs, batch_size=batch_size,
                                   num_saved_models=num_saved_models, overwrite_save=overwrite_save)


if __name__ == "__main__":
    main()
