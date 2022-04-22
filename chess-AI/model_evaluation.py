"""Helper file for evaluating quality of our learned models.

These tests help us decide which models to keep during learning.
"""
import chess
import torch
import random

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


class EasyStockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine.
    """
    def __init__(self, elo_rating=1000):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

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
    def __init__(self, elo_rating=1800):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

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


def testModel(num_games=5):
    wins = [0, 0, 0]
    player = Knightr0Player()
    enemyAI = [RandomPlayer(), EasyStockfishPlayer(), HardStockfishPlayer()]
    for _ in range(num_games):
        white = False
        for i in range(len(enemyAI)):
            white = not(white)
            board = chess.Board()
            if white:
                while True:
                    policy,_ = player.model(get_cnn_input(board))
                    board.push_uci(policy_converter.find_best_legal_move(policy, board))
                    if board.is_game_over():
                        print(board)
                        print()
                        if board.is_checkmate():
                            wins[i] += 1 
                        break
                    
                    board.push(enemyAI[i].select_move(board))
                    if board.is_game_over():
                        print(board)
                        print()
                        break
                    
            else:
                while True:
                    board.push(enemyAI[i].select_move(board))
                    if board.is_game_over():
                        print(board)
                        print()
                        break
                    
                    policy,_ = player.model(get_cnn_input(board))
                    board.push_uci(policy_converter.find_best_legal_move(policy, board))
                    if board.is_game_over():
                        print(board)
                        print()
                        if board.is_checkmate():
                            wins[i] += 1
                        break
                
    print(f"Num wins vs random: {wins[0]}/{num_games}")
    print("Win rate vs random: " + str(wins[0] / num_games))
    print(f"Num wins vs easy: {wins[1]}/{num_games}")
    print("Win rate vs easy: " + str(wins[1] / num_games))
    print(f"Num wins vs hard: {wins[2]}/{num_games}")
    print("Win rate vs hard: " + str(wins[2] / num_games))


def evaluate_two_players(player1, player2):
    board = chess.Board()


def main():
    testModel(50)

if __name__ == "__main__":
    main() 

