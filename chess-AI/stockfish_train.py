# User defined classes
from output_representation import PlayNetworkPolicyConverter
#from streamlit_dashboard import StreamlitDashboard
from model import PlayNetwork
from model_test import ModelTest

# Pychess and Stockfish engines
from stockfish import Stockfish
import chess

# Utilities 
import numpy as np
import time
import random
import streamlit as st


class StockfishTrain:

    '''
    Class for training our model utilizing stockfish
    '''

    def __init__(self):
        self.stockfish = Stockfish("/usr/local/bin/stockfish")
        self.policy_converter = PlayNetworkPolicyConverter()
        self.dashboard = StreamlitDashboard()


    def set_stockfish_params(self):
        elo,depth = self.dashboard.configure_stockfish()
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_depth(depth)

    def sig(self,value):
        return 1 / (1 + np.exp(value))

    def choose_move(self,moves):

        '''
        This ensures that we do not always just choose the top move
        Dependent on epsilon , it will either pick the top moe or 
        a random move
        '''
        epsilon = 0.3 
        e = np.random.rand() 

        if e < epsilon:
            move = moves[0]
        else:
            move = moves[random.randint(0,len(moves)-1)]
        return move

    def update_dataset_stats(self,dataset_stats,board,player,move_count):

        '''
        Will display the information about the dataset
        '''

        if board.is_stalemate():
            dataset_stats["stalemates"] += 1 
        else:
            if(player == 1):
                dataset_stats["black_wins"] = 5
            else:
                dataset_stats["white_wins"] += 1

        dataset_stats["completed_games"] += 1 
        dataset_stats["game_moves"].append(move_count)
        return dataset_stats

 
    def build_dataset(self,num_moves):

        '''
        Will build the data set of with a 
        size of num_moves
        '''

        move_probs = []
        move_values = []
        fen_strings = []

        # Dictionary to hold all the stats about the dataset
        dataset_stats = {
            "stalemates":0,
            "black_wins":0,
            "white_wins":0,
            "game_moves":[],
            "completed_games":0
        }

        move_count = 0

        fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen_string)
        player = 1

        start = time.time()

        for _ in range(int(num_moves)):
            # If the game is over then restart from the beginning, to build more examples
            if board.is_game_over():
                self.update_dataset_stats(dataset_stats,board,player,move_count)
                #Reset board
                fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                board = chess.Board(fen_string)
                player = 1
                continue

            move = self.simulate_move(board,fen_strings,move_probs,move_values,fen_string,player)
            move = chess.Move.from_uci(move) 
            move_count += 1
            board.push(move)

            # Alternate between white and black 1 = white, -1 = black
            player = player * -1
            fen_string = board.fen()
              
        end = time.time()

        dataset_stats["time"] = end - start
        self.dashboard.visualize_dataset(num_moves,dataset_stats)
        return fen_strings,move_probs,move_values


    def simulate_move(self,board,fen_strings,mcts_probs,mcts_evals,fen_string,player):

        '''
        This will build a dataset to be trained on
        '''

        self.stockfish.set_fen_position(fen_string)          

        # Stockfish always evaluates from whites perspective, this will also consider black
        stockfish_value = self.stockfish.get_evaluation()["value"] * player 


        # Will calculate the sigmoid and adjust from [-1 -> 1]
        stockfish_value = 1 - 2 * self.sig(stockfish_value)

        # Will get the top moves
        stockfish_topmoves = self.stockfish.get_top_moves(len(list(board.legal_moves)))
    
        moves = []
        search_probs = []

        # Flag for determining if best move has been found
        bestmove = False

        for move in stockfish_topmoves:
        
            # Get the move and append to the list of moves
            curr_move = move["Move"]
            moves.append(curr_move)


            # Want to set the search prob of the best move equal to 1 and the others equal to 0
            if (bestmove == False):
                search_probs.append(1)
                bestmove = True
            else:
                search_probs.append(0)
        
        full_search_probs = self.policy_converter.compute_full_search_probs(moves,
                                                        search_probs,
                                                        board
                                                        )

        # Appends the necessary values to the lists to be used for training
        fen_strings.append(fen_string)
        mcts_probs.append(full_search_probs)
        mcts_evals.append(stockfish_value)

        return self.choose_move(moves)
