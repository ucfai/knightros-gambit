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
