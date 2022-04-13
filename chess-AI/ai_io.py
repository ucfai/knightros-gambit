"""Module for loading/saving models/datasets, as well as initializing training parameters.
"""

import argparse
from datetime import datetime
import json
import os

import torch

from figshare_api import FigshareApi
import options
from streamlit_dashboard import Dashboard

figshare_api = FigshareApi()

def file_from_path(path):
    """Splits up the full file path into the directory and file."""
    base_path, file_name = os.path.split(path)
    return file_name

def create_date_string():
    """Gets a date string used for file naming."""
    curr_time = datetime.now()
    return curr_time.strftime("%d-%m-%y:%S")

def make_dir(dataset_path):
    """Create a directory for the corresponding dataset path if it does not already exist."""
    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

def load_dataset(ds_saving, show_dash):
    """Load a dataset from Figshare or locally.

    If figshare_load is true then the file name for load_path
    will be loaded into the path.
    """
    # See if dataset should be loaded from Figshare
    if ds_saving['figshare_load']:
        # Get the base path and file name
        file_name = file_from_path(ds_saving['load_path'])
        # ensure file exists in figshare
        assert figshare_api.get_figshare_article(ds_saving['load_path'], file_name), \
               "File not found in figshare"
        msg = f"{file_name} downloaded from figshare"
    else:
        # Load the dataset at the provided path. If using streamlit,
        # show a confirmation message.
        assert os.path.exists(ds_saving['load_path']), "Dataset not found at path provided." \
               "You must provide a path to an existing dataset"
        msg = "Dataset retrieved."
    if show_dash:
        Dashboard.info_message("success", msg)
    else:
        print(msg)
    # load the dataset into the path specified
    return torch.load(ds_saving['load_path'])

def save_dataset(dataset, ds_saving):
    """ Save a dataset to Figshare or Locally

    The save_path refers to the directory where the file
    will be saved. For saving to Figshare, the file still needs to
    be saved locally first
    """
    # make sure a path was specified
    if ds_saving['save_path'] is None:
        return
        
    date_string = create_date_string()
    # get the full file path to save
    full_path = ds_saving['save_path'] + "dataset-" +  date_string + ".pt"
    # save the dataset
    torch.save(dataset, full_path)

    if ds_saving['figshare_save']:
        # Save dataset to figshare
        title = "dataset-" + date_string
        desc = "This description is a placeholder"
        keys = ["Dataset", "Chess"]
        # 179 is category [Artificial Intelligence and Image Processing]
        categories = [179]
        figshare_api.upload(title, desc, keys, categories, full_path)

def save_model(nnet, m_saving, checkpointing, file_name=None):
    """Save a model to Figshare or Locally.

    The save_path refers to the directory where the file will be saved. For saving to Figshare,
    the file still needs to be saved locally first.
    """
    # A save path must be specified for the model to be saved
    if m_saving['save_path'] is not None:
        # Check if model is being saved as part of checkpointing
        if checkpointing:
            assert file_name is not None, " For checkpointing, a file name msut be specified"
            full_path = m_saving['checkpoint_path'] + file_name + ".pt"
            torch.save(nnet.state_dict(),full_path)
        else:
            date_string = create_date_string()
            full_path = m_saving['save_path'] + "model-" +  date_string + ".pt"
            torch.save(nnet.state_dict(), full_path)
            # Check if model should be saved to Figshare
            if m_saving['figshare_save']:
                title = "model-" + date_string
                desc = "This description is a placeholder"
                keys = ["Neural Network", "Chess"]
                # 179 is category, Artificial Intelligence and Image Processing
                categories = [179]
                api = FigshareApi()
                api.upload(title, desc, keys, categories, full_path)

def load_model(nnet, m_saving, show_dash):
    """Load model parameters from Figshare or locally."""
    # A load path must be specified to load a file
    if m_saving['load_path'] is not None:
        # Check if model should be loaded from Figshare
        if m_saving['figshare_load']:
            base_path,file_name = file_from_path(m_saving['load_path'])
            # Ensure file exists in figshare
            assert figshare_api.get_figshare_article(m_saving['load_path'], base_path, file_name), \
                   "File not found in figshare"
            msg = f"{file_name} downloaded from figshare"
        else:
            assert os.path.exists(m_saving['load_path']), "Model not found at path provided. " \
                   "You must provide a path to an existing model."
            msg = "Model retrieved."
        if show_dash:
            Dashboard.info_message("success", msg)
        else:
            print(msg)
        # load the model into memory
        nnet.load_state_dict(torch.load(m_saving['load_path'])) 

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
                        help='if specified, a dataset will be created')
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

        m_saving = params['saving']['m_saving']
        ds_saving = params['saving']['ds_saving']

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

        make_dataset = args.make_dataset
        stockfish_train = args.stockfish_train
        mcts_train = args.mcts_train

        # If using dashboard, this is set by the start button; set to True when reading from file
        start_train = True

    elif args.dashboard:
        dashboard = Dashboard()
        # TODO: Have reasonable defaults in case certain hyperparams are not specified within the
        # streamlit dashboard. Can use the params in params.json
        ds_saving, m_saving = dashboard.load_files()

        learning_rate, momentum, weight_decay = dashboard.nnet_params()

        elo, depth, stock_epochs, stock_games, stock_batch_size = dashboard.stockfish_params()

        mcts_epochs, mcts_batch_size, mcts_games, exploration, training_episodes, \
        mcts_simulations = dashboard.mcts_params()

        start_train = dashboard.train_button()

        make_dataset, stockfish_train, mcts_train = dashboard.train_flags()

    else:
        raise ValueError("This program must be run with the `train.sh` script. See the script "
                         "for usage instructions.")

    # want to wait until training has begun before model can be loaded
    if start_train and m_saving['load_path'] is not None:
        # if model is not coming from figshare then it must exist
        if not m_saving['figshare_load']:
            assert os.path.exists(m_saving['load_path']), "Model not found at path provided. " \
                   "To load a model you must provide a valid path."
        load_model(nnet, m_saving, args.dashboard)

    # add checkpoint folder property
    if m_saving['save_path'] is not None:
        m_saving["checkpoint_path"] = m_saving['save_path'] + "checkpoint-" + \
                                      create_date_string() + '/'
        make_dir(m_saving["checkpoint_path"])

    stockfish_options = options.StockfishOptions(learning_rate, momentum, weight_decay,
                                                 stock_epochs, stock_batch_size, stock_games,
                                                 device, m_saving, elo, depth)

    mcts_options = options.MCTSOptions(learning_rate, momentum, weight_decay, mcts_epochs,
                                       mcts_batch_size, mcts_games, device, m_saving,
                                       exploration, mcts_simulations, training_episodes)

    flags = options.TrainingFlags(
        start_train, args.dashboard, make_dataset, stockfish_train, mcts_train)
    return (nnet, ds_saving, stockfish_options, mcts_options, flags)

if __name__ == "__main__":
    print("no main for this file")
    