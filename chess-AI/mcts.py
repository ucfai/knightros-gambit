from math import sqrt

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
        self.n_values = {} 
        self.q_values = {}
        self.exploration = exploration 
        self.p_values = {}

    def set_current_qn_value(self, values, fen_string, uci_move, initial_value):
        if fen_string in values:
            if uci_move in values[fen_string]:
                return values[fen_string][uci_move]
            else:
                values[fen_string][uci_move] = initial_value
                return initial_value
        else:
            values[fen_string] = {}
            values[fen_string][uci_move] = initial_value
            return initial_value

    def find_best_move(self, legal_moves, fen_string):
        """Finds and returns the best move given the current state as a fen string and all legal moves.
        """
        best_u = -float("inf")
        best_move = -1

        for move in legal_moves:
            uci_move = move.uci()

            # Iterate over every legal move at current board state. Initialize q_value if state
            # has not been visited, otherwise get q_value.
            # if fen_string in self.q_values:
            #     if uci_move in self.q_values[fen_string]:
            #         q_value = self.q_values[fen_string][uci_move]
            #     else:
            #         self.q_values[fen_string][uci_move] = 0
            #         q_value = 0
            # else:
            #     self.q_values[fen_string] = {}
            #     self.q_values[fen_string][uci_move] = 0
            #     q_value = 0
            q_value = self.set_current_qn_value(self.q_values, fen_string, uci_move, initial_value=0)

            # TODO: Initial N value for unvisited nodes may need to be updated. Revisit this.

            # Iterate over every legal move at current board state. Initialize n_value if state
            # has not been visited, otherwise get n_value.
            # if fen_string in self.n_values:
            #     if uci_move in self.n_values[fen_string]:
            #         n_value = self.n_values[fen_string][uci_move]
            #     else:
            #         self.n_values[fen_string][uci_move] = 1
            #         n_value = float("inf")
            # else:
            #     self.n_values[fen_string] = {}
            #     self.n_values[fen_string][uci_move] = 1
            #     n_value = float("inf")
            n_value = self.set_current_qn_value(self.n_values, fen_string, uci_move, initial_value=1)

            # TODO: Look at different formulas for calculating U
            # There could be better formula for evaluating an action

            # Calculate U based on the UCB formula
            u_value = q_value + (self.exploration * self.p_values[fen_string][uci_move])
            u_value = u_value * sqrt(sum(self.n_values[fen_string].values())/n_value)
                        
            if u_value > best_u:
                best_u = u_value
                best_move = move  # Move with best u value

        return best_move
 
    def update_qn(self, fen_string, move, value):
        """Calculate q_values and n_values based on the move.
        """
        # TODO: Update this function to receive uci_move instead of creating it here.
        uci_move = move.uci()

        # Calculate q value for specified fen_string, uci_move, and value
        # TODO: Can look into different formulas for this as well
        self.q_values[fen_string][uci_move] = self.n_values[fen_string][uci_move] * self.q_values[fen_string][uci_move] + value
        self.q_values[fen_string][uci_move] = self.q_values[fen_string][uci_move] / (self.n_values[fen_string][uci_move] + 1)

        # Need to increment the number of times the node has been visited
        self.n_values[fen_string][uci_move] += 1

    def find_search_probs(self, fen_string):
        """Calculates and returns the search probabilites from the n_values for given fen_string.
        """
        # Calculates the search probabilites based on the the number of times a node has
        # been visited/sum of all nodes visited
        
        # TODO: Create correct formula for calculating search probs
        number_of_node_visits = np.array(list(self.n_values[fen_string].values()))
        search_probs = number_of_node_visits/np.sum(number_of_node_visits)

        # Return list of uci_moves and corresponding search probabilities
        return list(self.n_values[fen_string].keys()), search_probs

 
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

            # TODO: Get neural network state representation from fen_string
            # input_state = get_cnn_input(fen_string)

            # Return predictions and value from the nnet at the current state
            policy, value = nnet(get_cnn_input(board).float())
            policy = nnet.predict(policy, board)

            # Need to update P with the state and the policy
            self.p_values.update({fen_string: policy})

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
        self.update_qn(fen_string, move, value)

        # We return the negative value of the state, since alternate levels in the search tree
        # are from the perspective of different players. Since value \in [-1, 1], -value is the
        # value of the current board from the perspective of the other player.
        return -value

