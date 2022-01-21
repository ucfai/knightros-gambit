import numpy as np
import torch

import chess

# change input_states to include the past 8 moves and add 14 planes for each past move 
# (12 for pieces and 2 for repetition) to the cnn_input to make it 119 channels


def get_piece_planes(board2d, turn):
    piece_planes = np.zeros((12, 8, 8))
    w_piece_order = "PBNRQK"

    # orders the current player's pieces first
    if turn == 'w':
        piece_order = w_piece_order + w_piece_order.lower()
    else:
        piece_order = w_piece_order.lower() + w_piece_order

    for row in range(8):
        for col in range(8):
            plane_num = piece_order.find(board2d[row][col])
            if plane_num != -1:
                piece_planes[plane_num][row][col] = 1

    return piece_planes


def get_castle_planes(castling, turn):
    castling_planes = np.zeros((4, 8, 8))
    if turn == 'w':
        castle_order = "KQkq"
    else:
        castle_order = "kqKQ"

    for plane_num, char in enumerate(castle_order):
        if char in castling:
            castling_planes[plane_num].fill(1)

    return castling_planes


def get_turn_plane(turn):
    return np.zeros((1, 8, 8)) if turn == 'w' else np.ones((1, 8, 8))


def get_cnn_input(board_state):
    board_fen, turn, castling, _, half_count, move_count = board_state.fen().split(' ')
    board2d = fen_to_board(board_fen, turn)

    count_plane = np.full((1, 8, 8), round((float(move_count) - 1) / 74, 3))
    pointless_count = np.full((1, 8, 8), (int(half_count) / 2))

    cnn_input = torch.from_numpy(np.array([np.concatenate((get_piece_planes(board2d, turn), get_castle_planes(castling, turn),
                                                 get_turn_plane(turn), count_plane, pointless_count), axis=0)]))

    return cnn_input.float()


def fen_to_board(fen, turn):
    board_state = []

    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend(['-'] * int(c))
            else:
                brow.append(c)

        # flips perspective to current player
        if turn == 'w':
            board_state.append(brow)
        else:
            board_state.insert(0, brow)
    return board_state


def main():
    board_state = chess.Board()
    print(get_cnn_input(board_state).size())


if __name__ == '__main__':
    main()
