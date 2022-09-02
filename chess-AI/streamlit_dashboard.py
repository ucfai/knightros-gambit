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
        self.col1.title("Knightr0's Gambit")
        self.col2.title("Training Dashboard")

        self.col1.title("Get Dataset")
        self.col2.title("Get Model")

        # get directory for model and dataset
        data_dir = self.col1.text_input("Dataset Directory", "datasets/")
        model_dir = self.col2.text_input("Model Directory", "models/")

        # get the file names for each model / dataset
        ds_file_name = self.col1.text_input("Dataset file name", "dataset-06-04-22:19.pt")
        m_file_name = self.col2.text_input("Model file name", "model-06-04-22:19.pt")

        ds_load = self.col1.selectbox('Load Dataset', ('None', 'Locally', 'Figshare'))

        m_load = self.col2.selectbox('Load Model', ('None', 'Locally', 'Figshare'))

        self.col1.title("Save Dataset")
        self.col2.title("Save Model")

        # see if model should be saved to figshare
        ds_save_figshare = self.col1.checkbox("Save dataset to Figshare")
        m_save_figshare = self.col2.checkbox("Save model to Figshare")

        # get MCTS save frequency
        save_freq = self.col2.slider('MCTS Save freq', 1, 100)

        # get figshare and local load flags
        ds_figshare_load = (ds_load == 'Figshare')
        ds_local_load = (ds_load == 'Local')
        m_figshare_load = (m_load == 'Figshare')
        m_local_load = (m_load == 'Local')

        ds_saving = {
            "data_dir": data_dir,
            "file_name": ds_file_name,
            "figshare_load": ds_figshare_load,
            "local_load": ds_local_load,
            "figshare_save": ds_save_figshare
        }

        m_saving = {
            "model_dir": model_dir,
            "file_name": m_file_name,
            "figshare_load": m_figshare_load,
            "local_load": m_local_load,
            "figshare_save": m_save_figshare,
            "mcts_check_freq": save_freq
        }

        self.col1.subheader('Params')
        self.col1.code(ds_saving, language=None)
        self.col1.code(m_saving, language=None)

        return ds_saving, m_saving

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
        learning_rate = self.col2.number_input("Learning Rate", value=0.01)
        momentum = self.col2.number_input("Momentum", value=0.01)
        weight_decay = self.col2.number_input("Weight Decay", value=0.01)

        return learning_rate, momentum, weight_decay


    def train_flags(self):
        """ Gets the make dataset flag, stockfish train flag, and mcts train
        flag from the dashboard
        """
        self.col1.title("Training Flags")
        make_dataset_flag = self.col1.checkbox("Make Dataset", value=True)
        stockfish_train_flag = self.col1.checkbox("Stockfish Train", value=True)
        mcts_train_flag = self.col1.checkbox("MCTS Train", value=True)

        return make_dataset_flag, stockfish_train_flag, mcts_train_flag
        