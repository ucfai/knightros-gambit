from math import sqrt
import pandas as pd
import numpy as np
from state_representation import get_cnn_input
from output_representation import PlayNetworkPolicyConverter
import torch
import random


class Mcts:
    # Implementation of Monte Carlo Tree Search.
    #
    # Attributes:
    #     states_visited: List storing the states visited
    #     n_values: Dictionary storing the n values
    #     q_values: Dictionary storing the q values
    #     exploration: The exploration rate for the tree search
    #     p_values: Dictionary storing the search probabilites

    def __init__(self, exploration):
        self.states_visited = []
        self.exploration = exploration
        self.state_values = {}
        self.p_values = {}
        self.policy_converter = PlayNetworkPolicyConverter()

    # Get best possible move after running mcts_simulations number of simulations
    def get_best_move(self, mcts_simulations, board, nnet):
        # TODO: find_search_probs is unnecessary, as we only need to get the node
        # with the largest n_value rather than a probability distribution

        for _ in range(mcts_simulations):
            self.search(board, nnet)

        probs = self.find_search_probs(board.fen())

        # TODO: Consider using find_best_move instead of find_best_legal move
        best_move = self.policy_converter.find_best_legal_move(probs, board)

        return best_move

    # Returns q and n values if they exist, otherwise initializes them
    def set_current_qn_value(self, fen_string, uci_move):
        if fen_string in self.state_values:
            if uci_move in self.state_values[fen_string]:
                return self.state_values[fen_string][uci_move]
            else:
                self.state_values[fen_string][uci_move] = [0, 0]
                return 0, 0
        else:
            self.state_values[fen_string] = {}
            self.state_values[fen_string][uci_move] = [0, 0]
            return 0, 0

    # Finds and returns the best move given the current state as a fen string and all legal moves.
    def find_best_move(self, legal_moves, fen_string):
        best_u = -float("inf")
        best_move = -1

        for move in legal_moves:
            uci_move = move.uci()

            n_value, q_value = self.set_current_qn_value(fen_string, uci_move)
            values = self.state_values[fen_string].values()
            n_values = list(zip(*values))[0]

            # Calculate U based on the UCB formula
            u_value = q_value + (self.exploration * self.p_values[fen_string][uci_move]) * sqrt(sum(n_values)) / (1 + n_value)

            if u_value > best_u:
                best_u = u_value
                best_move = move  # Move with best u value

        return best_move

    # Update q_value and n_value
    def update_qn(self, fen_string, uci_move, value):
        n_value, q_value = self.state_values[fen_string][uci_move]

        if not isinstance(value, int):
            value = float(value.numpy())

        # Average new value into q_value and increment n_value for the new node visit
        self.state_values[fen_string][uci_move][1] = (n_value * q_value + value)/(n_value + 1)
        self.state_values[fen_string][uci_move][0] += 1

    # Calculates and returns the search probabilities from the n_values for given fen_string.
    def find_search_probs(self, fen_string, temperature):
        # TODO: Create correct formula for calculating search probs
        # Consider Dirchlet noise like in alpha zero for exploration

        values = self.state_values[fen_string].values()
        moves = list(self.state_values[fen_string].keys())
        n_values = np.array(list(zip(*values))[0])

        # Get flattened probability distribution
        n_values = torch.from_numpy(n_values)
        n_values = n_values ** (1 / temperature)
        search_probs = torch.nn.functional.softmax(n_values.float())
        move = random.choices(moves, search_probs)[0]

        # Return list of uci_moves and corresponding search probabilities
        return list(self.state_values[fen_string].keys()), search_probs, move

    # Method for performing a search on the tree.
    def search(self, board, nnet):
        fen_string = board.fen()

        # Assigns reward if game is over
        if board.is_game_over():
            return 1 if board.outcome().winner else 0

        # Checks to see if the node has been visited (expansion)
        if fen_string not in self.states_visited:
            self.states_visited.append(fen_string)

            # Get predictions and value from the nnet at the current state
            policy, value = nnet(get_cnn_input(board).float())
            policy = nnet.predict(policy, board)

            # Update P with network's policy output
            self.p_values.update({fen_string: policy})
            return -value

        # Select move to descend with
        move = self.find_best_move(board.legal_moves, fen_string)

        # Obtain backed up leaf node evaluation
        board.push(move)
        value = self.search(board, nnet)
        board.pop()

        # Update the Q and N values with the new evaluation
        self.update_qn(fen_string, move.uci(), value)

        # Back up value to previous nodes
        return -value
