"""
Main training program
"""
import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_io import init_params, load_model, save_model
from mcts import Mcts
from nn_layout import PlayNetwork
from output_representation import policy_converter
from state_representation import get_cnn_input
from stockfish_train import StockfishTrain
from streamlit_dashboard import Dashboard

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
            if game_num is not None and game_num % 10 == 0:  # Print every 10th game
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
    input_state = torch.stack(
        [get_cnn_input(chess.Board(state)) for game in game_data for state in game[0]])
    state_values = torch.tensor([state_val for game in game_data for state_val in game[1]]).float()
    move_probs = torch.tensor(
        np.array([move_prob for game in game_data for move_prob in game[2]])).float()

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
            Dashboard.info_message("success", "Epoch " + str(epoch) + " Finished")

    if show_dash:
        # Chart and show all the losses
        Dashboard.visualize_losses(average_pol_loss, average_val_loss)

    # Saves model to specified file, or a new file if not specified.
    # TODO: Figure frequency of model saving, right now it is after a defined number of epochs.
    save_model(nnet, options.save_path, options.num_saved_models, options.overwrite)


def train_on_stockfish(nnet, dataset_path, sf_opt, show_dash=False):
    stockfish = StockfishTrain(sf_opt.elo, sf_opt.depth)

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
            sf_opt.games, stocktrain_moves, stocktrain_value_approximator, show_dash)
        if show_dash:
            Dashboard.info_message("error", "No Dataset was found")
        # TODO: Dataset should be given a more descriptive name, this is just temporary
        # TODO: This should be in the ai_io file
        torch.save(dataset, './datasets/stockfish_data.pt')

    # Train using the stockfish dataset
    train_on_dataset(dataset, nnet, sf_opt)


def train_on_mcts(nnet, mcts_opt, show_dash=False):
    # Get MCTS object and parameters
    # TODO: Figure out how many times to perform self play (and better name for this variable)
    mcts = Mcts(mcts_opt.exploration, mcts_opt.device)

    # Will iterate through the number of training episodes
    for _ in range(mcts_opt.training_episodes):
        mcts_moves = lambda board: mcts.get_tree_results(mcts_opt.simulations, nnet, board, temperature=5)

        dataset = create_dataset(mcts_opt.games, mcts_moves)
        train_on_dataset(dataset, nnet, mcts_opt, show_dash)


def main():
    """Main function that will be run when starting training.
    """
    # Detect device to train on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nnet = PlayNetwork().to(device=device)

    nnet, dataset_path, stockfish_options, mcts_options, start_train, show_dash = init_params(nnet, device)

    if start_train:
        msg = "Stockfish Training Has Begun"
        if show_dash:
            Dashboard.info_message("success", msg)
        else:
            print(msg)
        train_on_stockfish(nnet, dataset_path, stockfish_options, show_dash)

        msg = "Stockfish Training completed"
        if show_dash:
            Dashboard.info_message("success", msg)
        else:
            print(msg)

        # Train network using MCTS
        msg = "MCTS Training has begun"
        if show_dash:
            Dashboard.info_message("success", msg)
        else:
            print(msg)
        train_on_mcts(nnet, mcts_options, show_dash)

        msg = "MCTS Training completed"
        if show_dash:
            Dashboard.info_message("success", msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()
