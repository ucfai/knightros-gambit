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
        self.w_pawns = np.zeros((8, 8))
        self.b_pawns = np.zeros((8, 8))
        self.w_bishs = np.zeros((8, 8))
        self.b_bishs = np.zeros((8, 8))
        self.w_knights = np.zeros((8, 8))
        self.b_knights = np.zeros((8, 8))
        self.w_rooks = np.zeros((8, 8))
        self.b_rooks = np.zeros((8, 8))
        self.w_queens = np.zeros((8, 8))
        self.b_queens = np.zeros((8, 8))
        self.w_king = np.zeros((8, 8))
        self.b_king = np.zeros((8, 8))
        self.bq_castle = np.zeros((8, 8))
        self.bk_castle = np.zeros((8, 8))
        self.wk_castle = np.zeros((8, 8))
        self.wq_castle = np.zeros((8, 8))
        self.turn_plane = np.zeros((8, 8))
        self.count_plane = np.full((8, 8), round((float(self.move_count)-1) / 74, 3))
        self.pointless_count = np.full((8, 8), (int(self.half_count) / 2))
        
    def set_state(self, board2d, fen, turn, castling, half_count, move_count):
        self.fen = fen
        self.turn = turn
        self.castling = castling
        self.half_count = half_count
        self.move_count = move_count
        self.board2d = board2d
        
        self.set_pieces()
        self.set_turn()
        self.set_castling()
        self.set_count()
        self.set_pointless()
    
    def set_pieces(self):
        for row in range(8):
            for col in range(8):
                if self.board2d[row][col] == 'p':
                    self.b_pawns[row][col] = 1
                elif self.board2d[row][col] == 'P':
                    self.w_pawns[row][col] = 1
                elif self.board2d[row][col] == 'b':
                    self.b_bishs[row][col] = 1
                elif self.board2d[row][col] == 'B':
                    self.w_bishs[row][col] = 1
                elif self.board2d[row][col] == 'n':
                    self.b_knights[row][col] = 1
                elif self.board2d[row][col] == 'N':
                    self.w_knights[row][col] = 1
                elif self.board2d[row][col] == 'r':
                    self.b_rooks[row][col] = 1
                elif self.board2d[row][col] == 'R':
                    self.w_rooks[row][col] = 1
                elif self.board2d[row][col] == 'q':
                    self.b_queens[row][col] = 1
                elif self.board2d[row][col] == 'Q':
                    self.w_queens[row][col] = 1
                elif self.board2d[row][col] == 'k':
                    self.b_king[row][col] = 1
                elif self.board2d[row][col] == 'K':
                    self.w_king[row][col] = 1
    
    def set_turn(self):
        if self.turn == "w":
            self.turn_plane.fill(0)
        else:
            self.turn_plane.fill(1)
            
    def set_castling(self):
        for char in self.castling:
            if char == 'K':
                self.wk_castle.fill(1)
            elif char == 'Q':
                self.wq_castle.fill(1)
            elif char == 'k':
                self.bk_castle.fill(1)
            elif char == 'q':
                self.bq_castle.fill(1)
                
    def set_count(self):
        self.count_plane.fill(round((float(self.move_count)-1)/74, 3))
    
    def set_pointless(self):
        self.pointless_count.fill(int(self.half_count)/2)
        
    def get_cnn_input(self):
        cnn_input = torch.from_numpy(np.array([self.w_pawns, self.b_pawns, self.w_bishs, self.b_bishs, self.w_knights, 
                                               self.b_knights, self.w_rooks, self.b_rooks, self.w_queens, self.b_queens,
                                               self.w_king, self.b_king, self.wk_castle, self.wq_castle, self.bk_castle, 
                                               self.bq_castle, self.turn_plane, self.count_plane, self.pointless_count])).reshape(1, 19, 8, 8)
        return cnn_input


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

# update_input updates the state of the cnn input to match the current state of the pychess board, 
# making the current game state passable into the cnn.
def update_input(board_state):
    board_fen, turn, castling, _, half_count, move_count = board_state.fen().split(' ')
    board2d = fen_to_board(board_fen)
    
    input_states.set_state(board2d, board_fen, turn, castling, half_count, move_count)


board_state = chess.Board()

# change input_states to include the past 8 moves and add 14 planes for each past move (12 for pieces and 2 for repetition)
# to the cnn_input to make it 119 channels

input_states = chess_state()
update_input(board_state)
print(input_states.get_cnn_input().size())
