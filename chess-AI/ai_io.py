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

class DatasetSaving:
    """Helper class containing the parameters used for saving and loading datasets.

    Attributes:
        data_dir: the directory where datasets will be saved and loaded from
        file_name: the file name of the dataset to load
        figshare_load: flag for if dataset should be loaded from figshare
        local_load: flag for if dataset should be loaded from local directory
        figshare_save: flag for if dataset should be saved to figshare
    """
    def __init__(self, data_dir, file_name, figshare_load, local_load, figshare_save, cp_freq):
        self.data_dir = data_dir
        self.file_name = file_name
        self.figshare_load = figshare_load
        self.local_load = local_load
        self.figshare_save = figshare_save
        self.cp_freq = cp_freq


class ModelSaving:
    """Helper class containing the parameters used for saving and loading models.

    Attributes:
        model_dir: the directory where models will be saved
        file_name: file name of the model to load
        figshare_load: flag for if model should be loaded from figshare
        local_load: flag for if model should be loaded from local directory
        figshare_save: flag for if model should be saved from figshare
        mcts_check_freq: frequency for how often model should be saved
        checkpoint_path: the path where models should be saved during checkpointing
    """
    def __init__(self, model_dir, file_name, figshare_load, local_load, figshare_save,
        mcts_check_freq, checkpoint_path=None):
        self.model_dir = model_dir
        self.file_name = file_name
        self.figshare_load = figshare_load
        self.local_load = local_load
        self.figshare_save = figshare_save
        self.mcts_check_freq = mcts_check_freq
        self.checkpoint_path = checkpoint_path


def file_from_path(path):
    """Splits up the full file path into the directory and file."""
    return os.path.split(path)[0]


def create_date_string():
    """Gets a date string used for file naming."""
    curr_time = datetime.now()
    return curr_time.strftime("%m-%d-%y_%H-%M-%S")


def make_dir(dataset_path):
    """Create a directory for the corresponding dataset path if it does not already exist."""
    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))


def load_dataset(dataset_saving, show_dash):
    """Load a dataset from Figshare or locally.

    If figshare_load is true then file specified under file_name
    will be loaded from figshareinto the directory provided by data_dir. If
    figshare_load is not true and local_load is true, the local file data_dir/file_name
    will be loaded. Otherwise no dataset will be loaded.
    """
    # See if dataset should be loaded from Figshare
    if dataset_saving.figshare_load:
        # Get the base path and file name
        file_name = dataset_saving.file_name
        # ensure file exists in figshare
        assert FigshareApi.get_figshare_article(dataset_saving.data_dir, file_name), \
               "File not found in figshare"
        msg = f"{file_name} downloaded from figshare"
    else:
        # Load the dataset at the provided path. If using streamlit,
        # show a confirmation message.
        assert os.path.exists(dataset_saving.data_dir), "Dataset not found at path provided." \
               "You must provide a path to an existing dataset"
        full_path = dataset_saving.data_dir + dataset_saving.file_name
        msg = f"dataset from {full_path} retrieved"

    if show_dash:
        Dashboard.info_message("success", msg)
    else:
        print(msg)
    # load the dataset into the path specified
    return torch.load(dataset_saving.data_dir + dataset_saving.file_name)


def save_dataset(dataset, data_dir, figshare_save=False, cp=None):
    """Save a dataset to Figshare or Locally

    The save_dir refers to the directory where the file
    will be saved. For saving to Figshare, the file still needs to
    be saved locally first.
    """
    date_string = create_date_string()
    # get the full file path to save
    name = date_string if cp is None else "{}-{}".format(date_string, cp)
    full_path = data_dir + "dataset-" + name + ".pt"
    # save the dataset
    torch.save(dataset, full_path)

    if figshare_save:
        # Save dataset to figshare
        title = "dataset-" + date_string
        desc = "This description is a placeholder"
        keys = ["Dataset", "Chess"]
        # 179 is category [Artificial Intelligence and Image Processing]
        categories = [179]
        FigshareApi.upload(title, desc, keys, categories, full_path)


def save_model(nnet, model_saving, checkpointing, file_name=None):
    """Save a model to Figshare or Locally.

    model_saving specifies the paths, file names, and flags for loading and
    saving nnet model weights. If figshare_save is true then the model
    will be saved to figshare, otherwise model will be saved locally.
    checkpoint_path specifies the path where the models from checkpointing will
    be saved. model_dir is the directory where models should be saved and loaded from.
    """
    if checkpointing:
        assert file_name is not None, "For checkpointing, a file name must be specified"
        full_path = model_saving.checkpoint_path + file_name + ".pt"
        torch.save(nnet.state_dict(), full_path)
    else:
        date_string = create_date_string()
        full_path = model_saving.model_dir + "model-" +  date_string + ".pt"
        torch.save(nnet.state_dict(), full_path)
        # Check if model should be saved to Figshare
        if model_saving.figshare_save:
            title = "model-" + date_string
            desc = "Dataset of chessboard states (fen strings), Stockfish move \
                probabilities, and position evaluations"
            keys = ["Neural Network", "Chess"]
            # 179 is category, Artificial Intelligence and Image Processing
            categories = [179]
            FigshareApi.upload(title, desc, keys, categories, full_path)


def load_model(nnet, model_saving, show_dash):
    """Load model parameters from Figshare or locally."""
    # A load path must be specified to load a file
    if model_saving.figshare_load:
        file_name = model_saving.file_name
        # Ensure file exists in figshare
        assert FigshareApi.get_figshare_article(model_saving.model_dir, file_name), \
                "File not found in figshare"
        msg = f"{file_name} downloaded from figshare"
    else:
        assert os.path.exists(model_saving.model_dir + model_saving.file_name), \
                "Model not found at path provided. " \
                "You must provide a path to an existing model."
        msg = "Model retrieved."
    if show_dash:
        Dashboard.info_message("success", msg)
    else:
        print(msg)
    # load the model into memory
    nnet.load_state_dict(torch.load(model_saving.model_dir + model_saving.file_name))


def init_params(nnet, device):
    """Initialize parameters used for training.

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
    """
    parser = argparse.ArgumentParser(
        description="Specifies whether to run train.py with streamlit or json. Note: if --json is "
                    "specified, it takes precedence over --dashboard.")
    parser.add_argument("-j", "--json",
                        dest="json",
                        action="store_true",
                        help="if specified, load params from file")
    parser.add_argument("-d", "--dashboard",
                        dest="dashboard",
                        action="store_true",
                        help="if specified, load params from dashboard")
    parser.add_argument("-m", "--make_dataset",
                        dest="make_dataset",
                        action="store_true",
                        help="if specified, a dataset will be created")
    parser.add_argument("-s", "--disable_stockfish",
                        dest="stockfish_train",
                        action="store_false",
                        help="if specified, stockfish train will be disabled")
    parser.add_argument("-t", "--disable_mcts",
                        dest="mcts_train",
                        action="store_false",
                        help="if specified, mcts train will be disabled")
    args = parser.parse_args()

    if args.json:
        with open("params.json") as file:
            params = json.load(file)

        model_saving_dict = params["saving"]["model_saving"]
        dataset_saving_dict = params["saving"]["dataset_saving"]

        learning_rate = params["misc_params"]["lr"]
        momentum = params["misc_params"]["momentum"]
        weight_decay = params["misc_params"]["weight_decay"]

        stock_epochs = params["stockfish"]["epochs"]
        stock_batch_size = params["stockfish"]["batch_size"]
        stock_games = params["stockfish"]["games"]
        elo, depth = params["stockfish"]["elo"], params["stockfish"]["depth"]

        mcts_epochs = params["mcts"]["epochs"]
        mcts_batch_size = params["mcts"]["batch_size"]
        mcts_games = params["mcts"]["games"]
        exploration = params["mcts"]["exploration"]
        training_episodes = params["mcts"]["training_episodes"]
        mcts_simulations = params["mcts"]["simulations"]

        make_dataset = args.make_dataset
        stockfish_train = args.stockfish_train
        mcts_train = args.mcts_train

        # If using dashboard, this is set by the start button; set to True when reading from file
        start_train = True

    elif args.dashboard:
        dashboard = Dashboard()
        # TODO: Have reasonable defaults in case certain hyperparams are not specified within the
        # streamlit dashboard. Can use the params in params.json
        dataset_saving_dict, model_saving_dict = dashboard.load_files()

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
        model_saving = ModelSaving(*model_saving_dict.values())
        dataset_saving = DatasetSaving(*dataset_saving_dict.values())

        if dataset_saving.data_dir is None:
            raise ValueError("Directory for dataset must always be specified")
        else:
            assert os.path.exists(os.path.dirname(dataset_saving.data_dir)), \
                "Dataset load directory does not exist"

        if model_saving.model_dir is None:
            raise ValueError("Directory for model must always be specified.")
        else:
            assert os.path.exists(os.path.dirname(model_saving.model_dir)), \
                "Model load directory does not exist"

        if dataset_saving.local_load or dataset_saving.figshare_load:
            if not dataset_saving.file_name:
                raise ValueError("A file name must be specified to load a dataset")
            if make_dataset:
                raise ValueError("Dataset cannot be created and loaded")
        elif not make_dataset:
            raise ValueError("If a dataset is not being created then local_load or \
                figshare_load must be specified")

        if model_saving.local_load or model_saving.figshare_load:
            if not model_saving.file_name:
                raise ValueError("A file name must be specified to load a model")
            if model_saving.local_load and model_saving.figshare_load:
                raise ValueError("Model cannot be loaded from figshare and locally")

        if dataset_saving.local_load and dataset_saving.figshare_load:
            raise ValueError("Dataset cannot be loaded from figshare and locally")

        # if no errors are raised then model can be loaded
        if model_saving.local_load or model_saving.figshare_load:
            load_model(nnet, model_saving, args.dashboard)

    # add checkpointing directory
    model_saving.checkpoint_path = model_saving.model_dir + "checkpoint-" + \
                                    create_date_string() + "/"
    make_dir(model_saving.checkpoint_path)

    stockfish_options = options.StockfishOptions(learning_rate, momentum, weight_decay,
                                                 stock_epochs, stock_batch_size, stock_games,
                                                 device, model_saving, elo, depth)

    mcts_options = options.MCTSOptions(learning_rate, momentum, weight_decay, mcts_epochs,
                                       mcts_batch_size, mcts_games, device, model_saving,
                                       exploration, mcts_simulations, training_episodes)

    flags = options.TrainingFlags(
        start_train, args.dashboard, make_dataset, stockfish_train, mcts_train)
    return (nnet, dataset_saving, stockfish_options, mcts_options, flags)

if __name__ == "__main__":
    print("no main for this file")
    