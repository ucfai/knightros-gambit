"""Helper file for evaluating quality of our learned models.

These tests help us decide which models to keep during learning.
"""
import chess
import torch
import random
import itertools
import numpy as np

from mcts import Mcts
from ai_io import init_params
from nn_layout import PlayNetwork
from state_representation import get_cnn_input
from output_representation import policy_converter
from util import create_stockfish_wrapper


class Knightr0Player:
    '''"AI" class that is a wrapper around our custom modification of AlphaZero.
    '''
    def __init__(self, path_to_model=None):
        if not(path_to_model == None):
            self.model = torch.load(path_to_model)
        else:
            self.model = PlayNetwork(testing=True).to('cpu')
        # TODO: Update exploration rate.
        self.mcts = Mcts(exploration=0.01, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def select_move(self, board):
        '''Sets stockfish state from provided fen and returns best move.
        '''
        return Mcts.get_best_move(board.fen(), self.model)

    def __str__(self):
        return "Knightr0Player"


class EasyStockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine.
    """
    def __init__(self, elo_rating=800):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)
        self.stockfish.depth = 5

    def select_move(self, board):
        """Sets stockfish state from provided fen and returns best move.
        """
        self.stockfish.set_fen_position(board.fen())
        return chess.Move.from_uci(self.stockfish.get_best_move())

    def __str__(self):
        return "EasyStockfishPlayer"


class HardStockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine.
    """
    
    def __init__(self, elo_rating=2000):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)
        self.stockfish.depth = 9
        self.elo = elo_rating

    def select_move(self, board):
        """Sets stockfish state from provided fen and returns best move.
        """
        self.stockfish.set_fen_position(board.fen())
        return chess.Move.from_uci(self.stockfish.get_best_move())

    def __str__(self):
        return "HardStockfishPlayer"


class RandomPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine, generating random moves.
    """
    def select_move(self, board):
        """Sets stockfish state from provided fen and returns random move.
        """
        # if type(move) is not str:
        #     move = move['Move']
        return random.choice(list(board.legal_moves))

    def __str__(self):
        return "RandomStockfishPlayer"
        

def evaluate_two_players(player1, player2, num_games=20):
    white = False
    pointsP1 = 0
    for _ in range(num_games):
        white = not(white)
        board = chess.Board()
        if white:
            while True:
                board = make_move(player1, board)
                if board.is_game_over(claim_draw=True):
                    if board.is_checkmate():
                        pointsP1 += 0.5
                    pointsP1 += 0.5
                    break
                
                board = make_move(player2, board)
                if board.is_game_over(claim_draw=True):
                    if not(board.is_checkmate()):
                        pointsP1 += 0.5
                    break        
        else:
            while True:
                board = make_move(player2, board)
                if board.is_game_over(claim_draw=True):
                    if not(board.is_checkmate()):
                        pointsP1 += 0.5
                    break
                    
                board = make_move(player1, board)
                if board.is_game_over(claim_draw=True):
                    if board.is_checkmate():
                        pointsP1 += 0.5
                    pointsP1 += 0.5
                    break
    pointsP2 = num_games - pointsP1
    return pointsP1, pointsP2

def make_move(player, board):
    if isinstance(player, Knightr0Player):
        policy,_ = player.model(get_cnn_input(board))
        board.push_uci(policy_converter.find_best_legal_move(policy, board))
    else:
        board.push(player.select_move(board))
    return board

def get_elo(points, num_games):
    expected = 0.5
    k_value = 32
    elos = np.zeros_like(points)
    length = len(points)
    for i in range(length):
        elos[i] = k_value * (points[i] - expected * num_games)
        
    maximum = max(elos)
    minimum = min(elos)
    for i in range(length):
        elos[i] = int(((elos[i] - minimum) / (maximum - minimum)) * (HardStockfishPlayer().elo))
    return elos

def main():
    num_games = 20
    players = [Knightr0Player(), RandomPlayer(), EasyStockfishPlayer(), HardStockfishPlayer()]
    tot_pts = np.zeros_like(players)
    matches = list(itertools.combinations(players, 2))
    pts = np.zeros_like(matches)
    length = len(matches)
    for i in range(length):
        pts[i][0], pts[i][1] = evaluate_two_players(matches[i][0], matches[i][1], num_games)
        for j in range(2):
            if isinstance(matches[i][j], Knightr0Player):
                tot_pts[0] += pts[i][j]
            elif isinstance(matches[i][j], RandomPlayer):
                tot_pts[1] += pts[i][j]
            elif isinstance(matches[i][j], EasyStockfishPlayer):
                tot_pts[2] += pts[i][j]
            elif isinstance(matches[i][j], HardStockfishPlayer):
                tot_pts[3] += pts[i][j]  

    elos = get_elo(tot_pts, num_games)
    print("\nRelative Elo")
    print("Knightr0:\t" + str(elos[0]))
    print("Random:\t\t" + str(elos[1]))
    print("Easy Stockfish:\t" + str(elos[2]))
    print("Hard Stockfish:\t" + str(elos[3]))

if __name__ == "__main__":
    main() 