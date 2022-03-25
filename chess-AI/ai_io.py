"""Module for loading/saving models/datasets, as well as initializing training parameters.
"""

import argparse
import json
import os

import torch

import options
from streamlit_dashboard import Dashboard


def save_model(nnet, save_path, num_saved_models, overwrite):
    """Save given model parameters to external file
    """
    if save_path is not None:
        torch.save(nnet.state_dict(), save_path)
    else:
        # Iterate through the number of models saved
        for i in range(num_saved_models):
            if not os.path.isfile(f'./models/models-{i + 1}.pt'):
                if overwrite and i != 0:
                    torch.save(nnet.state_dict(), f'./models/models-{i}.pt')
                    break
                torch.save(nnet.state_dict(), f'./models/models-{i + 1}.pt')
                break
            if i == num_saved_models - 1:
                torch.save(nnet.state_dict(), f'./models/models-{num_saved_models}.pt')


def load_model(nnet, model_path, num_saved_models):
    """Load model parameters into given network from external file
    """
    if model_path is not None:
        nnet.load_state_dict(torch.load(model_path))
    else:
        for i in range(num_saved_models):
            if not os.path.isfile(f'./models-{i + 2}.pt'):
                if i != 0:
                    nnet.load_state_dict(torch.load(f'./models-{i + 1}.pt'))
                break

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
        elo: Int, elo to use for stockfish engine
        depth: Int, depth to recurse when stockfish searches for best moves
        dataset_path: String, path to dataset to be loaded; can be None
        stockfish_options: TrainOptions, stores hyperparameters used for training
        exploration: Float, hyperparameter used to control amount of random choice during training
        training_episodes: Int, number of training episodes when running mcts
        mcts_simulations: Int, number of mcts simulations
        mcts_options: TrainOptions, stores hyperparameters used for training
        start_train: Bool, True by default if reading from file, else, set by train button on the
            streamlit dashboard.
        show_dash: Bool, specifies whether or not to show the dashboard.
    '''
    parser = argparse.ArgumentParser(
        description='Specifies whether to run train.py with streamlit or json. Note: if --json is '
                    'specified, it takes precedence over --dashboard.')
    parser.add_argument('-j', '--json',
                        dest='json',
                        action='store_true',
                        help='if specified, load params from file')
    parser.add_argument('-d', '--dashboard',
                        dest='dashboard',
                        action='store_true',
                        help='if specified, load params from dashboard')
    parser.add_argument('-m', '--make_dataset',
                        dest='make_dataset',
                        action='store_true',
                        help='if specified, a dataset will be made')
    parser.add_argument('-s', '--disable_stockfish',
                        dest='stockfish_train',
                        action='store_false',
                        help='if specified, stockfish train will be disabled')
    parser.add_argument('-t', '--disable_mcts',
                        dest='mcts_train',
                        action='store_false',
                        help='if specified, mcts train will be disabled')
    args = parser.parse_args()

    if args.json:
        with open('params.json') as file:
            params = json.load(file)

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

        make_dataset_flag = args.make_dataset
        stockfish_train_flag = args.stockfish_train
        mcts_train_flag = args.mcts_train

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

        make_dataset_flag, stockfish_train_flag, mcts_train_flag = dashboard.train_flags()

    else:
        raise ValueError("This program must be run with the `train.sh` script. See the script "
                         "for usage instructions.")

    # Load in a model
    if model_path is not None:
        load_model(nnet, model_path, num_saved_models)

    # Train network using stockfish evaluations
    stockfish_options = options.StockfishOptions(learning_rate, momentum, weight_decay, stock_epochs,
                                                 stock_batch_size, stock_games, device, model_path,
                                                 num_saved_models, overwrite, elo, depth)

    mcts_options = options.MCTSOptions(learning_rate, momentum, weight_decay, mcts_epochs,
                                       mcts_batch_size, mcts_games, device, model_path,
                                       num_saved_models, overwrite, exploration, mcts_simulations, training_episodes)

    return (nnet, dataset_path, stockfish_options, mcts_options, start_train, args.dashboard,
    make_dataset_flag, stockfish_train_flag, mcts_train_flag)

if __name__ == "__main__":
    print("no main for this file")
    