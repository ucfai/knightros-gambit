"""File for various Options classes used for hyperparameter encapsulation.
"""

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

    def __init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device,
                 save_path, num_saved_models, overwrite):
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


class MCTSOptions(TrainOptions):
    """Stores additional settings for training with MCTS

    Attributes:
        exploration: exploration constant for traversing the search tree
        simulations: number of iterations of MCTS to do before returning result
        training_episodes: how many times to generate and train on new dataset
    """

    def __init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device,
                 save_path, num_saved_models, overwrite, exploration, simulations, training_episodes):
        TrainOptions.__init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device,
                              save_path, num_saved_models, overwrite)

        self.exploration = exploration
        self.simulations = simulations
        self.training_episodes = training_episodes


class StockfishOptions(TrainOptions):
    """Stores additional settings for training with Stockfish

        Attributes:
            elo: elo for stockfish evals
            depth: search depth for stockfish evals
    """

    def __init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device,
                 save_path, num_saved_models, overwrite, elo, depth):
        TrainOptions.__init__(self, learning_rate, momentum, weight_decay, epochs, batch_size, games, device,
                              save_path, num_saved_models, overwrite)

        self.elo = elo
        self.depth = depth
