#https://web.stanford.edu/~surag/posts/alphazero.html

##from math import random
from math import sqrt
import numpy as np
import chess
from nn_layout import predict
import random
from state_representation import chess_state

class Mcts:
    
    def __init__(self):
        self.states_visited = [] # stores fen strings of states visisted
        self.N = {} # stores all the n values
        self.Q = {} # stores all the Q values
        self.exploration = .5 # stores the exploration values
        self.P = {} # stores the Porbabilites return by NN

    def find_best_action(self,legal_moves,fen_string):

        best_u = -float("inf")
        best_action = -1 
  
        for action in legal_moves: # finds the valid actions from current state

            uci_action = action.uci() # needs UCI format

            if fen_string in self.Q:
                if uci_action in self.Q[fen_string]:
                     q_value = self.Q[fen_string][uci_action] # get q value
                else:
                     self.Q[fen_string][uci_action] = 0
                     q_value = 0
            else:
                self.Q[fen_string]={}
                self.Q[fen_string][uci_action] = 0
                q_value = 0

            if fen_string in self.N:
                if uci_action in self.N[fen_string]:
                     n_value = self.N[fen_string][uci_action] # get q value
                else:
                     self.N[fen_string][uci_action] = 1
                     n_value = 1
            else:
                self.N[fen_string]={}
                self.N[fen_string][uci_action] = 1
                n_value = 1

            u = q_value  + self.exploration * self.P[fen_string][uci_action] * sqrt(sum(self.N[fen_string].values())/n_value)
   
            if u > best_u:
                best_action = action # action with best u value

        return best_action

    def update_QN(self,fen_string,action,value):

        action = action.uci()
        self.Q[fen_string][action] = (self.N[fen_string][action]* self.Q[fen_string][action] + value) / (self.N[fen_string][action] + 1)
        self.N[fen_string][action] += 1  
           
    def search(self,board,nnet): ## will find all the Q,N, and F values for a given simulation

        fen_string = board.fen()
        
        if board.is_game_over(): 
            print(board.outcome().termination)
            return  1 
       
        if fen_string not in self.states_visited: # if a state has not been visited then you must find the predictions made by the model
            self.states_visited.append(fen_string) # adds ot the fen string
            state_obj = chess_state(fen_string)
            input = state_obj.get_cnn_input()
            value,policy = predict(board,input)  # eventaul will pas input from intput representation
            self.P.update({fen_string:policy})
            return -value 
        
        legal_moves = board.legal_moves # gets the legal actions from the current position

        action = self.find_best_action(legal_moves,fen_string)
        board.push(action) # want to take the the best action
        value = self.search(board,nnet) # gets the value of the the action
        board.pop()
        fen_string = board.fen()
        self.update_QN(fen_string,action,value)
        
        return value

    def findSearchProbs(self,fen_string): # computes search probs for n
        values = np.array(list(self.N[fen_string].values()))    
        return values/np.sum(values)

def main():
    mcts = Mcts()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    for _ in range(50):
        board = chess.Board(start_fen)
        mcts.search(board,None)
    mcts.findSearchProbs(board.fen())

if __name__ == "__main__":
    main()