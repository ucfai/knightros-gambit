import torch
import chess
import numpy as np

class chess_state():
    def __init__(self):
        self.fen = ""
        self.turn = ''
        self.castling = ""
        self.move_count = 0
        self.half_count = 0
        self.board2d = np.array([])

        self.piece_planes = np.zeros((12, 8, 8))
        self.castling_planes = np.zeros((4, 8, 8))
        self.turn_plane = np.zeros((1, 8, 8))
        self.count_plane = np.full((1, 8, 8), round((float(self.move_count)-1) / 74, 3))
        self.pointless_count = np.full((1, 8, 8), (int(self.half_count) / 2))
        
    def set_state(self, board2d, fen, turn, castling, half_count, move_count):
        self.fen = fen
        self.turn = turn
        self.castling = castling
        self.half_count = half_count
        self.move_count = move_count
        self.board2d = board2d

        self.set_turn()
        self.set_pieces()
        self.set_castling()
        self.set_count()
        self.set_pointless()
    
    def set_pieces(self):
        w_piece_order = "PBNRQK"

        # orders the current player's pieces first
        if self.turn == 'w':
            piece_order = w_piece_order + w_piece_order.lower()
        else:
            piece_order = w_piece_order.lower() + w_piece_order

        for row in range(8):
            for col in range(8):
                plane_num = piece_order.find(self.board2d[row][col])
                if plane_num != -1:
                    self.piece_planes[plane_num][row][col] = 1
            
    def set_castling(self):
        if self.turn == 'w':
            castle_order = "KQkq"
        else:
            castle_order = "kqKQ"

        for plane_num, char in enumerate(castle_order):
            if char in self.castling:
                self.castling_planes[plane_num].fill(1)

    def set_turn(self):
        if self.turn == "w":
            self.turn_plane.fill(0)
        else:
            self.turn_plane.fill(1)
                
    def set_count(self):
        self.count_plane.fill(round((float(self.move_count)-1)/74, 3))
    
    def set_pointless(self):
        self.pointless_count.fill(int(self.half_count)/2)
        
    def get_cnn_input(self):
        cnn_input = torch.from_numpy(np.concatenate((self.piece_planes, self.castling_planes,
                                                     self.turn_plane, self.count_plane, self.pointless_count), axis=0))
        return cnn_input


def fen_to_board(fen, turn):
    board_state = []

    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( ['-'] * int(c) )
            else:
                brow.append(c)

        # flips perspective to current player
        if turn == 'w':
            board_state.append(brow)
        else:
            board_state.insert(0, brow)
    return board_state

# update_input updates the state of the cnn input to match the current state of the pychess board, 
# making the current game state passable into the cnn.
def update_input(board_state):
    board_fen, turn, castling, _, half_count, move_count = board_state.fen().split(' ')
    board2d = fen_to_board(board_fen, turn)
    
    input_states.set_state(board2d, board_fen, turn, castling, half_count, move_count)


board_state = chess.Board()

# change input_states to include the past 8 moves and add 14 planes for each past move (12 for pieces and 2 for repetition)
# to the cnn_input to make it 119 channels

input_states = chess_state()
update_input(board_state)
print(input_states.get_cnn_input().size())
