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
    dead_piece_graveyards["wq"] = [BoardCell(0, 9), BoardCell(0, 7), BoardCell(1, 7)]
    dead_piece_graveyards["wb"] = [BoardCell(1, 9), BoardCell(0, 6), BoardCell(1, 6)]
    dead_piece_graveyards["wn"] = [BoardCell(0, 8), BoardCell(0, 5), BoardCell(1, 5)]
    dead_piece_graveyards["wr"] = [BoardCell(1, 8), BoardCell(0, 4), BoardCell(1, 4)]
    dead_piece_graveyards["wp"] = [
        BoardCell(0, 0), BoardCell(1, 0), BoardCell(0, 1), BoardCell(1, 1),
        BoardCell(0, 2), BoardCell(1, 2), BoardCell(0, 3), BoardCell(1, 3),
    ]

    dead_piece_graveyards["bq"] = [BoardCell(10, 9), BoardCell(10, 7), BoardCell(11, 7)]
    dead_piece_graveyards["bb"] = [BoardCell(11, 9), BoardCell(10, 6), BoardCell(11, 6)]
    dead_piece_graveyards["bn"] = [BoardCell(10, 8), BoardCell(10, 5), BoardCell(11, 5)]
    dead_piece_graveyards["br"] = [BoardCell(11, 8), BoardCell(10, 4), BoardCell(11, 4)]
    dead_piece_graveyards["bp"] = [
        BoardCell(10, 0), BoardCell(11, 0), BoardCell(10, 1), BoardCell(11, 1),
        BoardCell(10, 2), BoardCell(11, 2), BoardCell(10, 3), BoardCell(11, 3),
    ]

    return dead_piece_graveyards

def init_capture_squares():
    '''Creates and returns two BoardCell objects corresponding to capture squares.
    '''
    w_capture_sq = BoardCell(1, 10)
    b_capture_sq = BoardCell(10, 10)

    return w_capture_sq, b_capture_sq
