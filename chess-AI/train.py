"""Main training program.
"""
import os

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_io import init_params, save_model, save_dataset, load_dataset, make_dir
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


def create_dataset(games, move_approximator, val_approximator=None, show_dash=False, cp_freq=None, data_dir=None):
    """Builds a dataset with the size of (games)

    Attributes:
        games: The total number of games to generate for the dataset
        val_approximator: Supplies a value for a given board state
        move_approximator: Supplies a policy distribution over legal moves and a move to take
    """

    assert data_dir is None or (data_dir is not None and cp_freq is not None), "data_dir can only be specified if cp on"
    # Storing gradients for all forward passes in each training game is demanding. Instead,
    # ignore gradients for now and store only the gradients needed for a particular batch later on
    game_data = []
    last_path = None

    with torch.no_grad():
        for i in range(games):
            if show_dash:
                game_data.append(training_game(val_approximator, move_approximator, game_num=i))
            else:
                game_data.append(training_game(val_approximator, move_approximator))

            if cp_freq is not None and i % cp_freq == 0 or i == games - 1:
                # Convert all the fen strings into tensors that are used in the dataset
                input_state = torch.stack(
                    [get_cnn_input(chess.Board(state)) for game in game_data for state in game[0]])
                state_values = torch.tensor([state_val for game in game_data for state_val in game[1]]).float()
                move_probs = torch.tensor(
                    np.array([move_prob for game in game_data for move_prob in game[2]])).float()

                # Create iterable dataset from game data
                dataset = TensorDataset(input_state, state_values, move_probs)

                if last_path is not None:
                    os.remove(last_path)

                last_path = save_dataset(dataset, data_dir, cp=i)

    # Return the dataset to be used
    return dataset


def train_on_dataset(dataset, nnet, options, iteration, save=True, show_dash=False, cp_freq=None):
    """Train with the specified dataset

    Attributes:
        dataset: the dataset to use for training
        nnet: the neural network
        options: Instance of TrainOptions containing settings/hyperparameters for training
    """
    last_path = None

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
    opt = torch.optim.SGD(nnet.parameters(), lr=options.learning_rate,
                          weight_decay=options.weight_decay, momentum=options.momentum)
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
            # TODO: Can maybe replace loop with 1 nnet call on inputs, then zip to separate lists
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

        if cp_freq is not None and epoch % cp_freq == 0:
            if last_path is not None:
                os.remove(last_path)
            last_path = save_model(nnet, options.model_saving, checkpointing=True, file_name=f"model_epoch{epoch}")

    if show_dash:
        # Chart and show all the losses
        Dashboard.visualize_losses(average_pol_loss, average_val_loss)

    # Saves model to specified file, or a new file if not specified.
    if save:
        save_model(nnet, options.model_saving, checkpointing=True, file_name=f"model_{iteration}")


def create_stockfish_dataset(sf_opt, dataset_saving, show_dash):
    """Create dataset of values/moves using stockfish

    Attributes:
        sf_opt: The stockfish training options
        show_dash: If true display info on dashboard
    """

    stockfish = StockfishTrain(sf_opt.elo, sf_opt.depth)

    # Value and move approximators from stockfish
    stocktrain_value_approximator = stockfish.get_value
    stocktrain_moves = lambda board: stockfish.get_move_probs(board, epsilon=0.3)

    return create_dataset(sf_opt.games, stocktrain_moves,stocktrain_value_approximator, show_dash,
                          cp_freq=dataset_saving.cp_freq, data_dir=dataset_saving.data_dir)


def train_on_mcts(nnet, mcts_opt, show_dash=False):
    """Use MCTS to improve value and move output of network

    Attributes:
        nnet: Network to be trained
        mcts_opt: Options for MCTS training
        show_dash: If true display info on dashboard
    """

    mcts = Mcts(mcts_opt.exploration, mcts_opt.device)

    # Will iterate through the number of training episodes
    for i in range(mcts_opt.training_episodes):
        mcts_moves = lambda board: mcts.get_tree_results(mcts_opt.simulations, nnet, board,
                                                         temperature=5)

        dataset = create_dataset(mcts_opt.games, mcts_moves)
        train_on_dataset(dataset, nnet, mcts_opt, iteration=(i+1),
                         save=(i % mcts_opt.model_saving.mcts_check_freq == 0), show_dash=show_dash)


def main():
    """Main function that will be run when starting training.
    """
    # Detect device to train on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nnet = PlayNetwork().to(device=device)

    nnet, dataset_saving, stockfish_options, mcts_options, flags = init_params(nnet, device)

    if flags.start_train:
        if flags.make_dataset:
            msg = "Dataset Creation Has Begun"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)
            dataset = create_stockfish_dataset(stockfish_options, dataset_saving, flags.show_dash)
            make_dir(dataset_saving.data_dir)
            save_dataset(dataset, dataset_saving.data_dir, dataset_saving.figshare_save)
            msg = "Dataset Creation completed"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)
        else:
            if dataset_saving.local_load or dataset_saving.figshare_load:
                dataset = load_dataset(dataset_saving, flags.show_dash)
            # load path must be specified
            else:
                raise ValueError("If not making dataset either local load or figshare \
                load must be specified")
        if flags.stockfish_train:
            msg = "Stockfish Training Has Begun"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)
            train_on_dataset(dataset, nnet, stockfish_options, iteration=0, cp_freq=stockfish_options.model_saving.stock_check_freq)

            msg = "Stockfish Training completed"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)

        if flags.mcts_train:
            # Train network using MCTS
            msg = "MCTS Training has begun"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)
            train_on_mcts(nnet, mcts_options, flags.show_dash)

            msg = "MCTS Training completed"
            if flags.show_dash:
                Dashboard.info_message("success", msg)
            else:
                print(msg)

        if mcts_options.model_saving.model_dir is not None:
            save_model(nnet, mcts_options.model_saving, checkpointing=False)


if __name__ == "__main__":
    main()
