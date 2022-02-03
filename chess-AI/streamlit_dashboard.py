import streamlit as st
import platform

class StreamlitDashboard:

    def train_button(self):

        return st.sidebar.button("Begin Training")

    def load_path(self):

        st.sidebar.title("Choose Model")
        file = st.sidebar.file_uploader("Model")
        if file:
            path = file.name
        else:
            path = None

        return path

    def load_dataset(self):

        st.sidebar.title("Choose dataset")
        file = st.sidebar.file_uploader("Dataset")
        if file:
            path = file.name
        else:
            path = None

        return path

    
    def set_stockfish_path(self):

        operating_system = platform.system().lower()
        if operating_system == "darwin":
            stockfish_path = "/usr/local/bin/stockfish"
        else:
            stockfish_path = st.text_input("Enter path where stockfish is located")

        return stockfish_path

    def set_mcts_params(self):

        st.sidebar.title("MCTS Params")
        mcts_simulations = st.sidebar.number_input("Number of MCTS simulations",value = 5)
        exploration = st.sidebar.number_input("Exploration rate ",value = 5)
        return mcts_simulations,exploration


    def set_training_data(self):
        st.sidebar.title("Stockfish Training Data")
        stocktrain_games = st.sidebar.number_input("Stockfish Training Games",value = 5)
        stocktrain_epochs = st.sidebar.number_input("Stockfish Training Epochs",value = 1)

        st.sidebar.title("Self Play Training Data")
        mcts_games = st.sidebar.number_input("Self Play Training Games ",value = 5)
        mcts_epochs = st.sidebar.number_input("Self Play Epochs",value = 1)

        return stocktrain_games,stocktrain_epochs,mcts_games,mcts_epochs
      
    def configure_stockfish(self):

        st.sidebar.title("Configure Stockfish")
        elo = st.sidebar.number_input("Elo rating",value = 1000)
        depth = st.sidebar.number_input("Depth",value = 2)

        return elo,depth
  
    def configure_dataset(self):
        st.sidebar.title("Dataset Size")
        num_moves = st.sidebar.number_input("Number of moves",value = 1000)
        return num_moves

    def set_nnet_hyperparamters(self):

        st.sidebar.title("NNET Hyperparameters")
      
        batch_size = st.sidebar.number_input("Batch Size",value = 8)
        lr = st.sidebar.number_input("Learning Rate",value = 0.1)

        return batch_size,lr

    def configure_nnet_structure(self):

        st.sidebar.title("Tweak Neural Network")

        res_blocks = st.sidebar.number_input("Residual Blocks",value = 40)
        filters = st.sidebar.number_input("Filters",value = 32)

        return res_blocks,filters

    def test_stockfish(self):

        st.sidebar.title("Test Against Stockfish")

        test_elo = st.sidebar.number_input("Test Elo",value = 1000)
        test_depth = st.sidebar.number_input("Test Depth",value = 5)
        mcts_simulations = st.sidebar.number_input("Mcts simulations",value = 100)
        num_games = st.sidebar.number_input("Number of games",value = 5)
        mcts_exploration = st.sidebar.number_input("Exploration for monte carlo",value = 0.1)

    def visualize_dataset(self,num_moves,dataset_stats):

        st.title("Dataset Stats")
        st.write("%d Training Examples generated in %f seconds!" % (num_moves, dataset_stats["time"]))
        st.write("Completed Games ", dataset_stats["completed_games"])
        st.write("Moves per game", dataset_stats["game_moves"])
        st.write("Stalemates ", dataset_stats["stalemates"])
        st.write("White Wins ", dataset_stats["white_wins"])
        st.write("Black Wins ", dataset_stats["black_wins"])
      

    def visualize_epochs(self,policy_loss,value_loss,end,start,num_moves,e):

        st.write("-----------------------------------------------------------")
        st.write("EPOCH %d --> [Average Loss Policy: %f ]" % (e,policy_loss)) 
        st.write("EPOCH %d --> [Average Loss Value: %f ]" % (e,value_loss))        
        st.write(" Time for epoch ", end - start)
        st.write(" Number of moves in epoch", num_moves)

        moves_second = (end-start)/num_moves

        st.write("Moves per (min,hour,day) = ( %0.1f , %0.1f , %0.1f ) "% (60 / moves_second , 3600 / moves_second, (3600 * 24) / moves_second ))


    def visualize_training_stats(self,average_pol_losses,average_val_losses):
        
        st.title("Training Stats")
        st.title("Policy Losses")
        st.line_chart(data = average_pol_losses)
        st.title("Value Losses")
        st.line_chart(data = average_val_losses )


    def visualize_test_stats(self):

            st.title("Testing against Stockfish")
            st.write("Total AI Wins", ai_wins)
            st.write("Total Stockfish Wins", stockfish_wins)
            st.write("Total Stalemates", stalemates) 