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

def init_capture_squares():
    '''Creates and returns two BoardCell objects corresponding to capture squares.
    '''
    w_capture_sq = BoardCell(1, 1)
    b_capture_sq = BoardCell(1, 10)

    return w_capture_sq, b_capture_sq

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
