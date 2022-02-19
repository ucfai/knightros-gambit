import streamlit as st


class Dashboard:
    '''
    Encapsulates all the data necessary for the 
    StreamtLit Dashboard
    '''

    def __init__(self):
        
        # Sets the page structure
        self.col1,self.col2 = st.columns(2)
        self.col3,self.col4 = st.columns(2)

    @staticmethod
    def info_message(type,msg):
        '''
        Displays different types of info messages

        args:
        type = type of message to be displayed
        msg = message to be displayed
        '''
        if type == "success":
            st.success(msg)
        elif type == "error":
            st.error(msg)
        elif type == "info":
            st.info(msg)
  
    @staticmethod
    def visualize_losses(pol_losses,val_losses):
        '''
        Will display charts and show loss stats
        '''
        st.title("Average Losses")

        st.write("Policy Losses")
        st.line_chart(pol_losses)
        st.metric(label="Loss", value=str(pol_losses[-1]), delta=str(pol_losses[-1] - pol_losses[0]))

        st.write("Value Losses")
        st.line_chart(val_losses)
        st.metric(label="Loss", value=str(val_losses[-1]), delta=str(val_losses[-1] - val_losses[0]))

    def train_button(self): 
        '''
        Just the button that is used to start training
        '''
        return st.button("Begin Training!")

    def load_files(self):
        '''
        Allows user to upload a dataset and a model 
        through the dashboard
        '''
        self.col1.title("Streamlit Training Dashboard")

        self.col1.title("Upload Dataset")
        dataset = self.col1.file_uploader("Dataset")

        self.col1.title("Upload Model")
        model = self.col1.file_uploader("Model")

        dataset_path = dataset.name if dataset else None
        model_path = model.name if model else None

        return dataset_path,model_path

    def stockfish_params(self):
        '''
        Sets all the necessary paramaters for stockfish
        '''
        self.col4.title("Stockfish Data")
        elo = self.col4.number_input("Elo rating", value=1000)
        depth = self.col4.number_input("Depth", value=2)
        games = self.col4.number_input("Games", value=1000)
        epochs = self.col4.number_input("Epochs", value=10)
        batch_size = self.col4.number_input("Batch Size", value=16)

        return elo, depth,epochs,games, batch_size

    
    def mcts_params(self):
        '''
        Sets all the MCTS paramaters
        '''
        self.col3.title("MCTS Parameters")
        epochs = self.col3.number_input("MCTS Epochs", value=10)
        batch_size = self.col3.number_input("MCTS Batch Size", value=16)
        games = self.col3.number_input("MCTS Games", value=1000)
        exploration = self.col3.number_input("MCTS Exploration", value=0.02)
        episodes = self.col3.number_input("MCTS Training Episodes", value=100)
        simulations = self.col3.number_input("MCTS Simulations", value=100)

        return epochs,batch_size,games,exploration,episodes,simulations

    
    def nnet_params(self):
        '''
        Sets all the NNET Hyperparameters
        '''
        self.col2.title("NNET Hyperparameters")
        saved_models = self.col2.number_input("Number of Saved Models", value=0)
        overwrite = self.col2.checkbox("Overwrite")
        lr = self.col2.number_input("Learning Rate", value=0.01)
        momentum = self.col2.number_input("Momentum", value=0.01)
        weight_decay = self.col2.number_input("Weight Decay", value=0.01)

        return saved_models,overwrite,lr,momentum,weight_decay
