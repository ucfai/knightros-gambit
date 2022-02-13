"""
Main training program
"""
import os
import time

import chess
import torch
from torch.utils.data import DataLoader, TensorDataset

from mcts import Mcts
from ai_io import save_model, load_model
from nn_layout import PlayNetwork
from output_representation import policy_converter
from state_representation import get_cnn_input
from stockfish_train import StockfishTrain


class TrainOptions:
    """This class stores settings for the training episodes"""

    def __init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device, save_path, num_saved_models, overwrite_save):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.games = games
        self.device = device
        self.save_path = save_path
        self.num_saved_models = num_saved_models
        self.overwrite_save = overwrite_save


def training_game(val_approximator, move_approximator):
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


def create_dataset(games, move_approximator, val_approximator=None):
    """Builds a dataset with the size of (games)

    Attributes:
        games: The total number of games to generate for the dataset
        val_approximator: Supplies a value for a given board state
        move_approximator: Supplies a policy distribution over legal moves and a move to take
    """

    # Storing gradients for all forward passes in each training game is demanding. Instead,
    # ignore gradients for now and store only the gradients needed for a particular batch later on
    with torch.no_grad():
        # Obtain data from training games
        game_data = [training_game(val_approximator, move_approximator) for _ in range(games)]

    # Convert all the fen strings into tensors that are used in the dataset
    input_state = torch.stack([get_cnn_input(chess.Board(state)) for game in game_data for state in game[0]])
    state_values = torch.tensor([state_val for game in game_data for state_val in game[1]]).float()
    move_probs = torch.tensor([move_prob for game in game_data for move_prob in game[2]]).float()

    # Create iterable dataset from game data
    dataset = TensorDataset(input_state, state_values, move_probs)

    # Return the dataset to be used
    return dataset


def train_on_dataset(dataset, nnet, options):
    """Train with the specified dataset

    Attributes:
        dataset: the dataset to use for training
        nnet: the neural network
        learning_rate: the learning rate for the optimizer
        momentum: additional parameter for the optimizer
        weight_decay: loss term encouraging smaller weights
        epochs: number of epochs
        batch_size: the batch size for SGD
        device: the device being used to train (either CPU or GPU)
        save_path: path for model checkpointing
        num_saved_models: number of models to store
    """

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

        end = time.time()

        # Calculate and store the average losses
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)
        average_pol_loss.append(policy_loss.cpu().detach().numpy())
        average_val_loss.append(value_loss.cpu().detach().numpy())

    # Saves model to specified file, or a new file if not specified.
    # TODO: Figure frequency of model saving, right now it is after every epoch
    # TODO: Determine better names for the models when saving
    if options.save_path is not None:
        save_model(nnet, options.save_path)
    else:
        # Iterate through the number of models saved
        for i in range(options.num_saved_models):
            if not (os.path.isfile(f'models/models-{i + 1}.pt')):
                if options.overwrite_save and i != 0:
                    save_model(nnet, f'models/models-{i}.pt')
                    break
                save_model(nnet, f'models/models-{i + 1}.pt')
                break
            if i == options.num_saved_models - 1:
                save_model(nnet, f'models/models-{options.num_saved_models}.pt')


def main():
    """Main function that will be run when starting training
    """

    # Detect device to train on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    stockfish_options = TrainOptions(learning_rate=0.1, momentum=0.9, weight_decay=0.001, epochs=10, batch_size=8,
                                     games=1000, device=device, save_path=None, num_saved_models=5, overwrite_save=True)

    mcts_options = TrainOptions(learning_rate=0.1, momentum=0.9, weight_decay=0.001, epochs=100, batch_size=8,
                                games=100, device=device, save_path=None, num_saved_models=5, overwrite_save=True)

    mcts_amt, mcts_simulations, exploration = 5, 100, 0.5

    nnet = PlayNetwork().to(device=device)

    # Will get the paths to load models and datasets from
    model_path = None
    dataset_path = None

    # Load in a model
    if model_path is not None:
        nnet = load_model(nnet, model_path)
    else:
        for i in range(num_saved_models):
            if not (os.path.isfile(f'chess-AI/models-{i + 2}.pt')):
                if i != 0:
                    nnet = load_model(nnet, f'chess-AI/models-{i + 1}.pt')
                break

    # Gets stockfish training object, and sets parameters (elo,depth)
    stockfish_path = "../../chess-engine/stockfish_14.1_win_x64_avx2.exe"
    stockfish = StockfishTrain(stockfish_path, elo=1000, depth=3)

    # Value and move approximators from stockfish
    stocktrain_value_approximator = stockfish.get_value
    stocktrain_moves = lambda board: stockfish.get_move_probs(board, epsilon=0.3)

    # Dataset needs to be either created or loaded
    if dataset_path:
        dataset = torch.load(dataset_path)
    else:
        dataset = create_dataset(stockfish_options.games, stocktrain_moves, stocktrain_value_approximator)
        # NOTE: Dataset should be given a more descriptive name, this is just temporary
        torch.save(dataset, 'datasets/stockfish_data.pt')

    # Train using the stockfish dataset
    train_on_dataset(dataset, nnet, stockfish_options)

    # Get MCTS object and parameters
    # TODO: Figure out how many times to perform self play (and better name for this variable)
    mcts = Mcts(exploration)

    # Will iterate through the number of training episodes
    for _ in range(mcts_amt):
        mcts_moves = lambda board: mcts.get_tree_results(mcts_simulations, nnet, board, temperature=5)

        dataset = create_dataset(mcts_options.games, mcts_moves)
        train_on_dataset(dataset, nnet, mcts_options)


if __name__ == "__main__":
    main()
