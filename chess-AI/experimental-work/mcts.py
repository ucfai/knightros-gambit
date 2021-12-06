from math import sqrt
import numpy as np
from state_representation import chess_state

class Mcts:

    def __init__(self):
        self.states_visited = [] # stores fen strings of states visited
        self.n_values = {} # stores all the N values
        self.q_values = {} # stores all the Q values
        self.exploration = 1 # sets exploration rate
        self.p_values = {} # stores the probabilities computed by NN

    def find_best_move(self,legal_moves,fen_string):

        best_u = -float("inf")
        best_move = -1

        # Iterate through all legal moves at the state

        for move in legal_moves:

            # Get move in uci form
            uci_move = move.uci()

            # Check to see if the state has been added to Q
            if fen_string in self.q_values:
                if uci_move in self.q_values[fen_string]:
                    q_value = self.q_values[fen_string][uci_move]
                else:
                    self.q_values[fen_string][uci_move] = 0
                    q_value = 0
            else:
                self.q_values[fen_string]={}
                self.q_values[fen_string][uci_move] = 0
                q_value = 0


            # Check to see if state has been added to N
            if fen_string in self.n_values:
                if uci_move in self.n_values[fen_string]:
                    n_value = self.n_values[fen_string][uci_move]
                else:
                    self.n_values[fen_string][uci_move] = 1
                    n_value = float("inf")
            else:
                self.n_values[fen_string]={}
                self.n_values[fen_string][uci_move] = 1
                n_value = float("inf")

            """
            NOTE: still unsure about the n values
            not sure if I should set the n values to infinite if a node
            has not been visited
            Also do not know what N should be initial
            """

            # Calculate U based on the UCB formula
            u_value = q_value + (self.exploration * self.p_values[fen_string][uci_move])
            u_value = u_value * sqrt(sum(self.n_values[fen_string].values())/n_value)

            """
            NOTE: Look at different formulas for calculating U
            There could be better formula for evaluating an action
            """

            # Want to choose the move with the highest u value
            if u_value > best_u:
                best_u = u_value
                best_move = move # move with best u value

        return best_move


    # Function to calculate Q and N based on the value and the move
    def update_qn(self,fen_string,move,value):

        move = move.uci()

        # Function to calculate Q
        self.q_values[fen_string][move] = self.n_values[fen_string][move] * self.q_values[fen_string][move] + value
        self.q_values[fen_string][move] = self.q_values[fen_string][move] / (self.n_values[fen_string][move] + 1)

        """
        NOTE: Can look into different formulas for this as well
        """

        # Need to increment the number of times the node has been visited
        self.n_values[fen_string][move] += 1

    # Calculates the search probabilties using N
    def find_search_probs(self,fen_string):

        # Calculates the search probabilites
        search_probs = np.array(list(self.n_values[fen_string].values()))
        search_probs = search_probs/np.sum(search_probs)

        """
        NOTE: The search probabilites are being calculated based on the
        the number of times a node has been visited/sum of all nodes visited
        """

        # Gets the list of all the moves
        keys = list(self.n_values[fen_string].keys())

        return keys,search_probs

    # Main function to traverse the tree
    def search(self,board,nnet):

        # Need to get the fen string for the current state
        fen_string = board.fen()

        # Assigns reward if game is over
        if board.is_game_over():
            return 1 if board.outcome().winner else 0

        # Checks to see if the node has been visited
        if fen_string not in self.states_visited:

            # Has to add the state to the list of visited nodes
            self.states_visited.append(fen_string)

            # Needs to get neural network state representation from fen_string
            state_representation = chess_state(fen_string)

            # Gets the input from neural network
            input_state = state_representation.get_cnn_input()
            value,policy = nnet.predict(board,input_state)

            """
            NOTE: Right now the state representation is not complete
            and working with the neural network
            The input is being passed into predict but not being used
            """

            # Need to update P with the state and the policy
            self.p_values.update({fen_string:policy})

            return -value

        # Finds the legal moves from the current state
        legal_moves = board.legal_moves

        # Finds the best move from the current state
        move = self.find_best_move(legal_moves,fen_string)

        # Makes the move on the board
        board.push(move)

        # Search based on the new state
        value = self.search(board,nnet)

        # After search need to undo the move
        board.pop()

        # After undoing the move , the Q and N values need to be updated
        fen_string = board.fen()
        self.update_qn(fen_string,move,value)

        return -value
