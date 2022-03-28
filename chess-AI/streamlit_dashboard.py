"""Streamlit dashboard, used for model training and hyperparameter initialization.

Optionally, can be used to upload a model or dataset to be used.
"""

import streamlit as st


class Dashboard:
    """Encapsulates all the data necessary for the StreamtLit Dashboard.
    """
    def __init__(self):
        # Sets the page structure
        self.col1, self.col2 = st.columns(2)
        self.col3, self.col4 = st.columns(2)

    @staticmethod
    def info_message(msg_type, msg):
        """Displays different types of info (success/failure) messages.

        Attributes:
            msg_type: type of message to be displayed, success or failure
            msg: string, message to be displayed
        """
        if msg_type == "success":
            st.success(msg)
        elif msg_type == "error":
            st.error(msg)
        elif msg_type == "info":
            st.info(msg)
        else:
            raise ValueError(f"Received invalid type: {msg_type}")

    @staticmethod
    def visualize_losses(pol_losses, val_losses):
        """Displays charts and shows loss stats.
        """
        st.title("Average Losses")

        st.write("Policy Losses")
        st.line_chart(pol_losses)
        st.metric(
            label="Loss", value=str(pol_losses[-1]), delta=str(pol_losses[-1] - pol_losses[0]))

        st.write("Value Losses")
        st.line_chart(val_losses)
        st.metric(
            label="Loss", value=str(val_losses[-1]), delta=str(val_losses[-1] - val_losses[0]))

    def train_button(self):
        """Creates start button that is used to begin training.
        """
        return st.button("Begin Training!")

    def load_files(self):
        """Create buttons to upload a dataset and a model through the dashboard.
        """
        self.col1.title("Streamlit Training Dashboard")

        self.col1.title("Upload Dataset")
        dataset = self.col1.file_uploader("Dataset")

        self.col1.title("Upload Model")
        model = self.col1.file_uploader("Model")

        dataset_path = dataset.name if dataset else None
        model_path = model.name if model else None

        return dataset_path, model_path

    def stockfish_params(self):
        """Sets all the necessary parameters for stockfish.
        """
        self.col4.title("Stockfish Data")
        elo = self.col4.number_input("Elo rating", value=1000)
        depth = self.col4.number_input("Depth", value=2)
        games = self.col4.number_input("Games", value=2)
        epochs = self.col4.number_input("Epochs", value=2)
        batch_size = self.col4.number_input("Batch Size", value=16)

        return elo, depth, epochs, games, batch_size

    def mcts_params(self):
        """Sets all the MCTS parameters.
        """
        self.col3.title("MCTS Parameters")
        epochs = self.col3.number_input("MCTS Epochs", value=2)
        batch_size = self.col3.number_input("MCTS Batch Size", value=16)
        games = self.col3.number_input("MCTS Games", value=2)
        exploration = self.col3.number_input("MCTS Exploration", value=0.02)
        episodes = self.col3.number_input("MCTS Training Episodes", value=2)
        simulations = self.col3.number_input("MCTS Simulations", value=2)

        return epochs, batch_size, games, exploration, episodes, simulations

    def nnet_params(self):
        """Sets all the neural network hyperparameters.
        """
        self.col2.title("NNET Hyperparameters")
        saved_models = self.col2.number_input("Number of Saved Models", value=0)
        overwrite = self.col2.checkbox("Overwrite")
        learning_rate = self.col2.number_input("Learning Rate", value=0.01)
        momentum = self.col2.number_input("Momentum", value=0.01)
        weight_decay = self.col2.number_input("Weight Decay", value=0.01)

        return saved_models, overwrite, learning_rate, momentum, weight_decay


    def train_flags(self):
        """ Gets the make dataset flag, stockfish train flag, and mcts train 
        flag from the dashboard
        """
        self.col1.title("Training Flags")
        make_dataset_flag = self.col1.checkbox("Make Dataset", value=True)
        stockfish_train_flag = self.col1.checkbox("Stockfish Train", value=True)
        mcts_train_flag = self.col1.checkbox("MCTS Train", value=True)

        return make_dataset_flag, stockfish_train_flag, mcts_train_flag