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
    return os.path.split(path)[0]


def create_date_string():
    """Gets a date string used for file naming."""
    curr_time = datetime.now()
    return curr_time.strftime("%d-%m-%y_%H-%M-%S")


def make_dir(dataset_path):
    """Create a directory for the corresponding dataset path if it does not already exist."""
    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))


def load_dataset(ds_saving, show_dash):
    """Load a dataset from Figshare or locally.

    If figshare_load is true then the file name for load_dir
    will be loaded into the path.
    """
    # See if dataset should be loaded from Figshare
    if ds_saving['figshare_load']:
        # Get the base path and file name
        file_name = ds_saving['file_name']
        # ensure file exists in figshare
        assert figshare_api.get_figshare_article(ds_saving['data_dir'], file_name), \
               "File not found in figshare"
        msg = f"{file_name} downloaded from figshare"
    else:
        # Load the dataset at the provided path. If using streamlit,
        # show a confirmation message.
        assert os.path.exists(ds_saving['data_dir']), "Dataset not found at path provided." \
               "You must provide a path to an existing dataset"
        full_path = ds_saving['data_dir'] + ds_saving['file_name']
        msg = f"dataset from {full_path} retrieved"

    if show_dash:
        Dashboard.info_message("success", msg)
    else:
        print(msg)
    # load the dataset into the path specified
    return torch.load(ds_saving['data_dir'] + ds_saving['file_name'])


def save_dataset(dataset, ds_saving):
    """ Save a dataset to Figshare or Locally

    The save_dir refers to the directory where the file
    will be saved. For saving to Figshare, the file still needs to
    be saved locally first.
    """
    # make sure a path was specified
    date_string = create_date_string()
    # get the full file path to save
    full_path = ds_saving['data_dir'] + "dataset-" +  date_string + ".pt"
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

    The save_dir refers to the directory where the file will be saved. For saving to Figshare,
    the file still needs to be saved locally first.
    """

    if checkpointing:
        assert file_name is not None, " For checkpointing, a file name msut be specified"
        full_path = m_saving['checkpoint_path'] + file_name + ".pt"
        torch.save(nnet.state_dict(), full_path)
    else:
        date_string = create_date_string()
        full_path = m_saving['model_dir'] + "model-" +  date_string + ".pt"
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
    if m_saving['figshare_load']:
        file_name = m_saving['file_name']
        # Ensure file exists in figshare
        assert figshare_api.get_figshare_article(m_saving['model_dir'], file_name), \
                "File not found in figshare"
        msg = f"{file_name} downloaded from figshare"
    else:
        assert os.path.exists(m_saving['model_dir'] + m_saving['file_name']), \
                "Model not found at path provided. " \
                "You must provide a path to an existing model."
        msg = "Model retrieved."
    if show_dash:
        Dashboard.info_message("success", msg)
    else:
        print(msg)
    # load the model into memory
    nnet.load_state_dict(torch.load(m_saving['model_dir'] + m_saving['file_name']))


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

    # wait until training has begun and catch any param errors
    if start_train:
        if ds_saving["data_dir"] is None:
            raise ValueError("Directory for dataset must always be specified")
        else:
            assert os.path.exists(os.path.dirname(ds_saving["data_dir"])), \
                "Dataset load directory does not exist"

        if m_saving["model_dir"] is None:
            raise ValueError("Directory for model must always be specified.")
        else:
            assert os.path.exists(os.path.dirname(m_saving["model_dir"])), \
                "Model load directory does not exist"

        if ds_saving["local_load"] or ds_saving["figshare_load"]:
            if not ds_saving["file_name"]:
                raise ValueError("A file name must be specified to load a dataset")
            if make_dataset:
                raise ValueError("Dataset cannot be created and loaded")
        elif not make_dataset:
            raise ValueError("If a dataset is not being created then local_load or \
                figshare_load must be specified")

        if m_saving["local_load"] or m_saving["figshare_load"]:
            if not m_saving["file_name"]:
                raise ValueError("A file name must be specified to load a model")

        if ds_saving["local_load"] and ds_saving["figshare_load"]:
            raise ValueError("Dataset cannot be loaded from figshare and locally")

        if m_saving["local_load"] and m_saving["figshare_load"]:
            raise ValueError("Model cannot be loaded from figshare and locally")

        # if no errors are raised then model can be loaded
        if m_saving["local_load"] or m_saving["figshare_load"]:
            load_model(nnet, m_saving, args.dashboard)

    # add checkpointing directory
    m_saving["checkpoint_path"] = m_saving["model_dir"] + "checkpoint-" + \
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
    