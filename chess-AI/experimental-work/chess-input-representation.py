#Not the most efficient but puts the ideas of representation in an example program.

import numpy as np
import chess

def fen_to_board(fen):
    board_state = []
    w_play = True
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( ['-'] * int(c) )
            else:
                brow.append(c)

        board_state.append(brow)
    return board_state

def update_state(board_state):
    board_fen, turn, castling, _, half_count, move_count = board_state.fen().split(' ')
    board = fen_to_board(board_fen)

    w_pawns = np.zeros((8,8))
    b_pawns = np.zeros((8,8))
    w_bishs = np.zeros((8,8))
    b_bishs = np.zeros((8,8))
    w_knights = np.zeros((8,8))
    b_knights = np.zeros((8,8))
    w_rooks = np.zeros((8,8))
    b_rooks = np.zeros((8,8))
    w_queens = np.zeros((8,8))
    b_queens = np.zeros((8,8))
    w_king = np.zeros((8,8))
    b_king = np.zeros((8,8))
    bq_castle = np.zeros((8,8))
    bk_castle = np.zeros((8,8))
    wk_castle = np.zeros((8,8))
    wq_castle = np.zeros((8,8))

    if turn == 'w':
        turn_plane = np.zeros((8,8))
    else:
        turn_plane = np.ones((8,8))

    for char in castling:
        if char == 'K':
            wk_castle = np.ones((8,8))
        elif char == 'Q':
            wq_castle = np.ones((8,8))
        elif char == 'k':
            bk_castle = np.ones((8,8))
        elif char == 'q':
            bq_castle = np.ones((8,8))

    count_plane = np.full((8,8), round((float(move_count)-1)/74, 3))

    pointless_count = np.full((8,8), (int(half_count)/2))

    for row in range(8):
        for col in range(8):
            if(board[row][col] == 'p'):
                b_pawns[row][col] = 1
            elif(board[row][col] == 'P'):
                w_pawns[row][col] = 1
            elif(board[row][col] == 'b'):
                b_bishs[row][col] = 1
            elif(board[row][col] == 'B'):
                w_bishs[row][col] = 1
            elif(board[row][col] == 'n'):
                b_knights[row][col] = 1
            elif(board[row][col] == 'N'):
                w_knights[row][col] = 1
            elif(board[row][col] == 'r'):
                b_rooks[row][col] = 1
            elif(board[row][col] == 'R'):
                w_rooks[row][col] = 1
            elif(board[row][col] == 'q'):
                b_queens[row][col] = 1
            elif(board[row][col] == 'Q'):
                w_queens[row][col] = 1
            elif(board[row][col] == 'k'):
                b_king[row][col] = 1
            elif(board[row][col] == 'K'):
                w_king[row][col] = 1
    #probably better to use an array for return, will update just wanted to get it down.
    return board_fen, w_pawns, b_pawns, w_bishs, b_bishs, w_knights, b_knights, w_rooks, b_rooks, w_queens, b_queens, w_king, b_king, wk_castle, wq_castle, bk_castle, bq_castle, turn_plane, count_plane, pointless_count

board_state = chess.Board()
board_fen, w_pawns, b_pawns, w_bishs, b_bishs, w_knights, b_knights, w_rooks, b_rooks, w_queens, b_queens, w_king, b_king, wk_castle, wq_castle, bk_castle, bq_castle, turn_plane, count_plane, pointless_count = update_state(board_state)

##print(board_fen)
##print(pointless_count)
##print(count_plane)
##print(bk_castle)
##print(b_pawns)
##print(w_pawns)

