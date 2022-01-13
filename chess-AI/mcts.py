
from math import sqrt
import pandas as pd
import numpy as np
from state_representation import get_cnn_input

class Mcts:
    """Implementation of Monte Carlo Tree Search.

    Attributes:
        states_visited: List storing the states visited
        n_values: Dictionary storing the n values
        q_values: Dictionary storing the q values
        exploration: The exploration rate for the tree search
        p_values: Dictionary storing the search probabilites  
    """
    def __init__(self, exploration):
        self.states_visited = [] 
        self.exploration = exploration 
        self.state_values = {}
        self.p_values = {}

    def set_current_qn_value(self, fen_string, uci_move):

        if fen_string in self.state_values:
            if uci_move in self.state_values[fen_string]:
                return self.state_values[fen_string][uci_move]
            else:
                self.state_values[fen_string][uci_move] = [1,0]
                return (1,0)
        else:
            self.state_values[fen_string] = {}
            self.state_values[fen_string][uci_move] = [1,0]
            return (1,0)

    def find_best_move(self, legal_moves, fen_string):
        """Finds and returns the best move given the current state as a fen string and all legal moves.
        """
        best_u = -float("inf")
        best_move = -1

        for move in legal_moves:

            uci_move = move.uci()

            n_value,q_value = self.set_current_qn_value(fen_string, uci_move)
            
            # TODO: Look at different formulas for calculating U
            # There could be better formula for evaluating an action

            # Calculate U based on the UCB formula

            values = self.state_values[fen_string].values()
            n_values = list(zip(*values))[0]

            u_value = q_value + (self.exploration * self.p_values[fen_string][uci_move]) \
                      * sqrt(sum(n_values)/n_value)
                        
            if u_value > best_u:
                best_u = u_value
                best_move = move  # Move with best u value

        return best_move
 
    def update_qn(self, fen_string, uci_move, value):
        """Calculate q_values and n_values based on the move.
        """
        # TODO: Update this function to receive uci_move instead of creating it here.
     

        # Calculate q value for specified fen_string, uci_move, and value
        # TODO: Can look into different formulas for this as well

        n_value, q_value = self.state_values[fen_string][uci_move]

        self.state_values[fen_string][uci_move][1] = n_value * q_value + value
        self.state_values[fen_string][uci_move][1] = q_value/ (n_value + 1)

        # Need to increment the number of times the node has been visited
        self.state_values[fen_string][uci_move][0] += 1

    def find_search_probs(self, fen_string):
        """Calculates and returns the search probabilites from the n_values for given fen_string.
        """
        # Calculates the search probabilites based on the the number of times a node has
        # been visited/sum of all nodes visited
        
        # TODO: Create correct formula for calculating search probs

        values = self.state_values[fen_string].values()
        n_values = list(zip(*values))[0]

        number_of_node_visits = np.array(n_values)
        search_probs = number_of_node_visits/np.sum(number_of_node_visits)

        # Return list of uci_moves and corresponding search probabilities
        return list(self.state_values[fen_string].keys()), search_probs

    # @jit
    def search(self, board, nnet):
        """Method for performing a search on the tree.
        """
        fen_string = board.fen()
        # Assigns reward if game is over
        if board.is_game_over():
            return 1 if board.outcome().winner else 0

        # Checks to see if the node has been visited
        if fen_string not in self.states_visited:
            # Add the state to the list of visited nodes
            self.states_visited.append(fen_string)

            # Get predictions and value from the nnet at the current state
            policy, value = nnet(get_cnn_input(board).float())
            policy = nnet.predict(policy, board)

            # Need to update P with the state and the policy
            self.p_values.update({fen_string: policy})
            return value

        # Finds the best move from the current state
        move = self.find_best_move(board.legal_moves, fen_string)

        # Makes the move on the board
        board.push(move)

        # Search based on the new state
        value = self.search(board, nnet)

        # After search need to undo the move
        board.pop()

        # After undoing the move , the Q and N values need to be updated
        fen_string = board.fen()
        self.update_qn(fen_string, move.uci(), value)

        # Returns value to previous nodes
        return value

