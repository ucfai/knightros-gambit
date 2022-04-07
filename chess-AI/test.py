import chess

from random import choice
from nn_layout import PlayNetwork
from output_representation import policy_converter
from state_representation import get_cnn_input
from ai_io import init_params

def main(num_games=20):
    nnet = PlayNetwork(testing=True).to('cpu')
    nnet, _, _, _, _ = init_params(nnet, 'cpu')
    wins = 0
    white = False
    for _ in range(num_games):
        white = not(white)
        board = chess.Board()
        if white:
            while True:
                policy,_ = nnet(get_cnn_input(board))
                board.push_uci(policy_converter.find_best_legal_move(policy, board))
                if board.is_game_over():
                    if board.is_checkmate():
                        wins += 1 
                    break
                
                board.push(choice(list(board.legal_moves)))
                if board.is_game_over():
                    break
                
        else:
            while True:
                board.push(choice(list(board.legal_moves)))
                if board.is_game_over():
                    break

                policy,_ = nnet(get_cnn_input(board))
                board.push_uci(policy_converter.find_best_legal_move(policy, board))
                if board.is_game_over():
                    if board.is_checkmate():
                        wins += 1 
                    break
        print(board)
        print()
                
    print(f"Num wins: {wins}/{num_games}")
    print("Win rate: " + str(wins / num_games))
                
if __name__ == "__main__":
    main()


