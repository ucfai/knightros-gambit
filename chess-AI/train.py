"""
Main training program
"""
import argparse
import json

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_io import save_model, load_model
from mcts import Mcts
from nn_layout import PlayNetwork
from output_representation import policy_converter
from state_representation import get_cnn_input
from stockfish_train import StockfishTrain
from streamlit_dashboard import Dashboard


class TrainOptions:
    """Stores settings for the training episodes

    Attributes:
        learning_rate: the learning rate for the optimizer
        momentum: additional parameter for the optimizer
        weight_decay: loss term encouraging smaller weights
        epochs: number of epochs
        batch_size: the batch size for SGD
        games: number of games to run when creating dataset
        device: the device being used to train (either CPU or GPU)
        save_path: path for model checkpointing
        num_saved_models: number of models to store
    """

    def __init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device, save_path, num_saved_models, overwrite):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.games = games
        self.device = device
        self.save_path = save_path
        self.num_saved_models = num_saved_models
        self.overwrite = overwrite

def training_game(val_approximator, move_approximator, game_num=None):
    """Run a full game, storing fen_strings, policies, and values

    Attributes:
        val_approximator: Supplies a value for a given board state
        move_approximator: Supplies a policy distribution over legal moves and a move to take

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
        moves, move_probs, move = move_approximator(board)

        # Converts move search probabilites to (8,8,73) vector to be used for training
        move_probs_vector = policy_converter.compute_full_search_probs(moves, move_probs, board)
        all_move_probs.append(move_probs_vector)

        # val_approximator will not be none for Stockfish
        if val_approximator is not None:
            all_state_values.append(val_approximator(board))

        # Makes the move on the pychess board
        move = chess.Move.from_uci(move)
        board.push(move)

        # If the game is over, end the episode
        # TODO: Consider removing `board.can_claim_draw()` as it may be slow to check.
        # See https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
        if board.is_game_over() or board.can_claim_draw():
            if game_num is not None:
                # Print board and display message on streamlit dashboard
                print(board, end="\n\n")

                Dashboard.info_message("success", f"{game_num + 1} Games Generated")
            break

    # value_approximator will be none for mcts
    if val_approximator is None:
        all_state_values = assign_rewards(board, len(all_move_probs))

    return board_fens, all_state_values, all_move_probs


def assign_rewards(board, length):
    """Iterates through training examples and assigns rewards based on result of the game.
    NOTE: Want to triple check this logic, reward assignment is very important
    """
    reward = 0

    state_values = [0 for _ in range(length)]

    # if board.outcome() is None, the game is not over
    # if board.outcome().winner is None, the game is a draw
    if (board.outcome() is not None) and (board.outcome().winner is not None):
        reward = -1
    for move_num in range(length - 1, -1, -1):
        reward *= -1
        state_values[move_num] = reward

    return state_values

def create_dataset(games, move_approximator, val_approximator=None, show_dash=False):
    """Builds a dataset with the size of (games)

    Attributes:
        games: The total number of games to generate for the dataset
        val_approximator: Supplies a value for a given board state
        move_approximator: Supplies a policy distribution over legal moves and a move to take
    """

    # Storing gradients for all forward passes in each training game is demanding. Instead,
    # ignore gradients for now and store only the gradients needed for a particular batch later on
    with torch.no_grad():
        if show_dash:
            # Print every 10th game for visualization purposes
            game_data = [training_game(val_approximator, move_approximator,
                                       game_num=i) for i in range(games)]
        else:
            game_data = [training_game(val_approximator, move_approximator) for _ in range(games)]

    # Convert all the fen strings into tensors that are used in the dataset
    input_state = torch.stack([get_cnn_input(chess.Board(state)) for game in game_data for state in game[0]])
    state_values = torch.tensor([state_val for game in game_data for state_val in game[1]]).float()
    move_probs = torch.tensor(np.array([move_prob for game in game_data for move_prob in game[2]])).float()

    # Create iterable dataset from game data
    dataset = TensorDataset(input_state, state_values, move_probs)

    # Return the dataset to be used
    return dataset


def train_on_dataset(dataset, nnet, options, show_dash=False):
    """Train with the specified dataset

    Attributes:
        dataset: the dataset to use for training
        nnet: the neural network
        options: Instance of TrainOptions containing settings/hyperparameters for training
    """
    if show_dash:
        Dashboard.info_message("info", "Training on Dataset")

    # Stores the average losses which are used for graphing
    average_pol_loss = []
    average_val_loss = []

    # Define loss functions (cross entropy and mean squared error)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()

    # Create optimizer for updating parameters during training
    # TODO: Consider using other optimizers, such as Adam
    opt = torch.optim.SGD(nnet.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay, momentum=options.momentum)
    train_dl = DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=False)

    # Main training loop
    for epoch in range(options.epochs):

        # Variables used solely for monitoring training, not used for actually updating the model
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
                policy, value = nnet(state.to(device=options.device))
                policy_batch.append(policy)
                value_batch.append(value)

            # Convert the list of tensors to a single tensor for policy and value.
            policy_batch = torch.stack(policy_batch).float().to(options.device)
            value_batch = torch.stack(value_batch).flatten().float().to(options.device)

            move_probs = move_probs.to(device=options.device)
            state_values = state_values.to(device=options.device)

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

        # Calculate and store the average losses
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)
        average_pol_loss.append(policy_loss.cpu().detach().numpy())
        average_val_loss.append(value_loss.cpu().detach().numpy())

        if show_dash:
            # Keep track of when each epoch is over
            Dashboard.info_message("success","Epoch " + str(epoch) + " Finished" )

    if show_dash:
        # Chart and show all the losses
        Dashboard.visualize_losses(average_pol_loss,average_val_loss)

    # Saves model to specified file, or a new file if not specified.
    # TODO: Figure frequency of model saving, right now it is after a defined number of epochs.
    save_model(nnet, options.save_path, options.num_saved_models, options.overwrite)


def train_on_stockfish(nnet, elo, depth, dataset_path, options, show_dash=False):
    stockfish = StockfishTrain(elo, depth)
    # Value and move approximators from stockfish
    stocktrain_value_approximator = stockfish.get_value
    stocktrain_moves = lambda board: stockfish.get_move_probs(board, epsilon=0.3)

    # Dataset needs to be either created or loaded
    if dataset_path is not None:
        if show_dash:
            Dashboard.info_message("success", "Dataset Found!")
        dataset = torch.load(dataset_path)
    else:
        dataset = create_dataset(
            options.games, stocktrain_moves, stocktrain_value_approximator, show_dash)
        if show_dash:
            Dashboard.info_message("error", "No Dataset was found")
        # TODO: Dataset should be given a more descriptive name, this is just temporary
        # TODO: This should be in the ai_io file
        torch.save(dataset, './datasets/stockfish_data.pt')

    # Train using the stockfish dataset
    train_on_dataset(dataset, nnet, options)


def train_on_mcts(nnet, exploration, mcts_simulations, training_episodes, options, args):
    # Get MCTS object and parameters
    # TODO: Figure out how many times to perform self play (and better name for this variable)
    mcts = Mcts(exploration, options.device)

    # Will iterate through the number of training episodes
    for _ in range(training_episodes):
        mcts_moves = lambda board: mcts.get_tree_results(mcts_simulations, nnet, board, temperature=5)

        dataset = create_dataset(options.games, mcts_moves)
        train_on_dataset(dataset, nnet, options)

def init_params(nnet, device):
    '''Initialize parameters used for training.

    Depending on value of flags passed to program, either set parameters from json file or from
    the streamlit dashboard (see streamlit_dashboard.py).

    Attributes:
        nnet: Instance of neural net with randomly initialized weights; weights are loaded into
            this model from file specified by user.
        device: A context manager that specifies torch.device to use for training.
    Returns:
        nnet: Initialized neural network with weights loaded from file.
        elo: 
        depth: 
        dataset_path: 
        stockfish_options: 
        exploration: 
        training_episodes: 
        mcts_simulations: 
        mcts_options: 
        start_train: 
        show_dash: 
    '''
    # 
    parser = argparse.ArgumentParser(
        description='Specifies whether to run train.py with streamlit or json. Note: if --json is '
                    'specified, it takes precedence over --dashboard.')
    parser.add_argument('-j', '--json',
                        dest='json',
                        action='store_const',
                        const=True,
                        default=False,
                        help='if specified, load params from file')
    parser.add_argument('-d', '--dashboard',
                        dest='dashboard',
                        action='store_const',
                        const=True,
                        default=False,
                        help='if specified, load params from dashboard')
    args = parser.parse_args()

    if args.json:
        with open('params.json') as f:
            params = json.load(f)

        model_path = params['saving']['model_path']
        dataset_path = params['saving']['dataset_path']

        num_saved_models = params['saving']['num_saved_models']
        overwrite = params['saving']['overwrite']
        learning_rate = params['misc_params']['lr']
        momentum = params['misc_params']['momentum']
        weight_decay = params['misc_params']['weight_decay']

        stock_epochs = params['stockfish']['epochs']
        stock_batch_size = params['stockfish']['batch_size']
        stock_games = params['stockfish']['games']
        elo, depth = params['stockfish']['elo'], params['stockfish']['depth']

        mcts_epochs = params['mcts']['epochs']
        mcts_batch_size = params['mcts']['batch_size']
        mcts_games = params['mcts']['games'] 
        exploration = params['mcts']['exploration']
        training_episodes = params['mcts']['training_episodes']
        mcts_simulations = params['mcts']['simulations']

        # If using dashboard, this is set by the start button; set to True when reading from file
        start_train = True

    elif args.dashboard:
        dashboard = Dashboard()
        # TODO: Have reasonable defaults in case certain hyperparams are not specified within the
        # streamlit dashboard. Can use the params in params.json
        model_path, dataset_path = dashboard.load_files()

        num_saved_models, overwrite, learning_rate, \
        momentum, weight_decay = dashboard.nnet_params()

        elo, depth, stock_epochs, stock_games, stock_batch_size = dashboard.stockfish_params()

        mcts_epochs, mcts_batch_size, mcts_games, exploration, training_episodes, \
        mcts_simulations = dashboard.mcts_params()

        start_train = dashboard.train_button()

    else:
        raise ValueError("This program must be run with the `train.sh` script. See the script "
                         "for usage instructions.")

    # Load in a model
    if model_path is not None:
        load_model(nnet, model_path, num_saved_models)

    # Train network using stockfish evaluations
    stockfish_options = TrainOptions(learning_rate, momentum, weight_decay, stock_epochs,
                                     stock_batch_size, stock_games, device, model_path,
                                     num_saved_models, overwrite)

    mcts_options = TrainOptions(learning_rate, momentum, weight_decay, mcts_epochs,
                                mcts_batch_size, mcts_games, device, model_path,
                                num_saved_models, overwrite)

    # TODO: Create MctsOptions and StockfishOptions to house these params that are related
    return (nnet, elo, depth, dataset_path, stockfish_options, exploration, training_episodes,
            mcts_simulations, mcts_options, start_train, args.dashboard)


def main():
    """Main function that will be run when starting training.
    """
    # Detect device to train on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nnet = PlayNetwork().to(device=device)

    # TODO: this looks bad and would be better if we encapsulate more of these params
    nnet, elo, depth, dataset_path, stockfish_options, exploration, training_episodes, \
        mcts_simulations, mcts_options, start_train, show_dash = init_params(nnet, device)

    if start_train:
        if show_dash:
            Dashboard.info_message("success", "Stockfish Training Has Begun")
        train_on_stockfish(nnet, elo, depth, dataset_path, stockfish_options, show_dash)

        if show_dash:
            Dashboard.info_message("success", f"Stockfish Training completed")

        # Train network using MCTS
        if show_dash:
            Dashboard.info_message("success", "MCTS Training has begun")
        train_on_mcts(
            nnet, exploration, training_episodes, mcts_simulations, mcts_options, show_dash)

        if show_dash:
            Dashboard.info_message("success", f"MCTS Training completed")


if __name__ == "__main__":
    main()
