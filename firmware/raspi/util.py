'''Helper file for miscellaneous utility classes and functions.
'''
import platform

from stockfish import Stockfish

class BoardCell:
    '''Helper class for indexing entirety of board.
    '''
    def __init__(self, row=None, col=None):
        '''Point, if no args passed, initialized to (0,0), top left corner of board

        Corresponds to center of board square
        '''
        self.row = row if row else 0
        self.col = col if col else 0

    def __str__(self):
        # TODO: Update this to have a clearer naming scheme for the indexing.
        return f"{chr(self.row+ord('a'))}{chr(self.col+ord('a'))}"

    def get_coords(self):
        '''Returns tuple of row and col indices.
        '''
        return (self.row, self.col)

    def to_chess_sq(self):
        return chr(self.col + ord('a')) + chr(self.row + ord('1'))

def create_stockfish_wrapper():
    '''Create simple wrapper around stockfish python module depending on operating system type.
    '''
    operating_system = platform.system().lower()
    if operating_system == "darwin":
        stockfish_path = "/usr/local/bin/stockfish"
    elif operating_system == "raspi":
        stockfish_path = "n/a"
    elif operating_system == "linux":
        stockfish_path = "../../chess-engine/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"
    elif operating_system == "windows":
        stockfish_path = "n/a"
    else:
        raise ValueError("Operating system must be one of "
                         "'darwin' (osx), 'linux', 'windows', 'raspi'")
    return Stockfish(stockfish_path)

def init_dead_piece_counts():
    '''Creates and returns a dictionary corresponding to number of dead pieces for each piece type.
    '''
    dead_piece_counts = {}
    dead_piece_counts["wq"] = 2
    dead_piece_counts["wb"] = 1
    dead_piece_counts["wn"] = 1
    dead_piece_counts["wr"] = 1
    dead_piece_counts["wp"] = 0

    dead_piece_counts["bq"] = 2
    dead_piece_counts["bb"] = 1
    dead_piece_counts["bn"] = 1
    dead_piece_counts["br"] = 1
    dead_piece_counts["bp"] = 0

    return dead_piece_counts

# TODO: Update all BoardCells to use 2D resolution
def init_dead_piece_graveyards():
    '''Creates and returns a dictionary of BoardCell for each dead piece type.
    '''
    dead_piece_graveyards = {}
    dead_piece_graveyards["wq"] = [BoardCell(2, 10), BoardCell(4, 10), BoardCell(4, 11)]
    dead_piece_graveyards["wb"] = [BoardCell(2, 11), BoardCell(5, 10), BoardCell(5, 11)]
    dead_piece_graveyards["wn"] = [BoardCell(3, 10), BoardCell(6, 10), BoardCell(6, 11)]
    dead_piece_graveyards["wr"] = [BoardCell(3, 11), BoardCell(7, 10), BoardCell(7, 11)]
    dead_piece_graveyards["wp"] = [
        BoardCell(8, 10), BoardCell(8, 11), BoardCell(9, 10), BoardCell(9, 11),
        BoardCell(10, 10), BoardCell(10, 11), BoardCell(11, 10), BoardCell(11, 11),
    ]

    dead_piece_graveyards["bq"] = [BoardCell(2, 0), BoardCell(4, 0), BoardCell(4, 1)]
    dead_piece_graveyards["bb"] = [BoardCell(2, 1), BoardCell(5, 0), BoardCell(5, 1)]
    dead_piece_graveyards["bn"] = [BoardCell(3, 0), BoardCell(6, 0), BoardCell(6, 1)]
    dead_piece_graveyards["br"] = [BoardCell(3, 1), BoardCell(7, 0), BoardCell(7, 1)]
    dead_piece_graveyards["bp"] = [
        BoardCell(8, 0), BoardCell(8, 1), BoardCell(9, 0), BoardCell(9, 1),
        BoardCell(10, 0), BoardCell(10, 1), BoardCell(11, 0), BoardCell(11, 1),
    ]

    return dead_piece_graveyards

# TODO: Update all BoardCells to use 2D resolution
def init_capture_squares():
    '''Creates and returns two BoardCell objects corresponding to capture squares.
    '''
    w_capture_sq = BoardCell(1, 1)
    b_capture_sq = BoardCell(1, 10)

    return w_capture_sq, b_capture_sq

def uci_move_from_boardcells(source, dest):
    return source.to_chess_sq() + dest.to_chess_sq()

def get_piece_info_from_square(square, grid):
    '''Returns tuple of color and piece type from provided square.
    '''
    coords = get_chess_coords_from_square(square)
    piece_w_color = grid[coords.row][coords.col]
    if piece_w_color == '.':
        return (None, None)
    color = 'w' if piece_w_color.isupper() else 'b'
    return (color, piece_w_color.lower())

def get_chess_coords_from_square(square):
    '''Converts chess square to a BoardCell.

    Example: a1 <=> [0, 0], h8 <=> [7, 7], regardless of whether human plays white or black pieces.
    '''
    # Nums correspond to row (rank), letters correspond to col (files)
    return BoardCell(ord(square[1]) - ord('1'), ord(square[0]) - ord('a'))

def get_2d_board(fen, turn=None):
    '''Returns a 2d board from fen representation

    Taken from this SO answer:
    https://stackoverflow.com/questions/66451525/how-to-convert-fen-id-onto-a-chess-board
    '''
    board = []
    for row in reversed(fen.split('/')):
        brow = []
        for char in row:
            if char == ' ':
                break
            if char in '12345678':
                brow.extend(['.'] * int(char))
            else:
                brow.append(char)
        board.append(brow)

    # flips perspective to current player
    if turn:
        if turn == 'w':
            board_state.append(brow)
        else:
            board_state.insert(0, brow)
    return board

def is_promotion(prev_board_fen, move):
    # Note: This differs from boardinterface.Engine.is_promotion in that it checks for a promotion
    # in the case that the UCI move is not yet known. It is less efficient as it creates a 2d grid
    # to check for the piece previously at the square, corresponding to move[:2], and thus should
    # only be used when boardinterface.Engine.is_promotion can not be used.

    # If piece in prev_board_fen at square move[:2] is a pawn and move[3] is the final rank,
    # this is a promotion. Note: Don't need to check color since white pawn can't move to row 1
    # and vice versa for black
    return (get_piece_info_from_square(move[:2], get_2d_board(prev_board_fen))[1] == 'p') and \
           (move[3] in ('1', '8'))
