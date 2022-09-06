"""Helper file for miscellaneous utility classes and functions.
"""
import argparse
import io
import os
import platform

import chess.pgn  # pylint: disable=import-error
from stockfish import Stockfish  # pylint: disable=import-error

class BoardCell:
    """Helper class for indexing entirety of board.
    """
    def __init__(self, row=None, col=None):
        """Point, if no args passed, initialized to (0,0), bottom left corner of entire board.

        Board representation uses two unit spaces for each cell. So center of bottom left cell is
        at (1, 1). Similarly, center of bottom left cell of chessboard (either 'a1' or 'h8'
        depending on whether human plays white or black) is (5, 5). The bottom left corner of the
        playing area of the chessboard is (4, 4). The top right corner of the entire board is at
        (24, 24).
        """
        self.row = row if row else 0
        self.col = col if col else 0

    def __str__(self):
        return f"{chr(self.row + ord('A'))}{chr(self.col + ord('A'))}"

    def get_coords(self):
        """Returns tuple of row and col indices.
        """
        return (self.row, self.col)

    def to_chess_sq(self):
        """Returns string representation of BoardCell.

        Assumes BoardCell(0, 0) <=> 'a1' and BoardCell(7, 7) <=> 'h8'.
        """
        return chr(self.col + ord('a')) + chr(self.row + ord('1'))

def create_stockfish_wrapper():
    """Create simple wrapper around stockfish python module depending on operating system type.
    """
    operating_system = platform.system().lower()
    if operating_system == "darwin":
        stockfish_path = "/usr/local/bin/stockfish"
    elif operating_system == "raspi":
        stockfish_path = "n/a"
    elif operating_system == "linux":
        stockfish_path = "../../chess-engine/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"
    elif operating_system == "windows":
        stockfish_path = "../../chess-engine/stockfish_14.1_win_x64_avx2.exe"
    else:
        raise ValueError("Operating system must be one of "
                         "'darwin' (osx), 'linux', 'windows', 'raspi'")
    return Stockfish(stockfish_path)

def init_dead_piece_counts():
    """Creates and returns a dictionary corresponding to number of dead pieces for each piece type.
    """
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
    """Creates and returns a dictionary of BoardCell for each dead piece type.
    """
    # TODO: Initialization of the graveyard depends on orientation of the board. If human plays
    # black pieces, this initialization needs to reflect that board configuration.
    dead_piece_graveyards = {}
    dead_piece_graveyards["wq"] = [BoardCell(4, 20), BoardCell(8, 20), BoardCell(8, 22)]
    dead_piece_graveyards["wb"] = [BoardCell(4, 22), BoardCell(10, 20), BoardCell(10, 22)]
    dead_piece_graveyards["wn"] = [BoardCell(6, 20), BoardCell(12, 20), BoardCell(12, 22)]
    dead_piece_graveyards["wr"] = [BoardCell(6, 22), BoardCell(14, 20), BoardCell(14, 22)]
    dead_piece_graveyards["wp"] = [
        BoardCell(16, 20), BoardCell(16, 22), BoardCell(18, 20), BoardCell(18, 22),
        BoardCell(20, 20), BoardCell(20, 22), BoardCell(22, 20), BoardCell(22, 22),
    ]

    dead_piece_graveyards["bq"] = [BoardCell(4, 0), BoardCell(8, 0), BoardCell(8, 2)]
    dead_piece_graveyards["bb"] = [BoardCell(4, 2), BoardCell(10, 0), BoardCell(10, 2)]
    dead_piece_graveyards["bn"] = [BoardCell(6, 0), BoardCell(12, 0), BoardCell(12, 2)]
    dead_piece_graveyards["br"] = [BoardCell(6, 2), BoardCell(14, 0), BoardCell(14, 2)]
    dead_piece_graveyards["bp"] = [
        BoardCell(16, 0), BoardCell(16, 2), BoardCell(18, 0), BoardCell(18, 2),
        BoardCell(20, 0), BoardCell(20, 2), BoardCell(22, 0), BoardCell(22, 2),
    ]

    return dead_piece_graveyards

def init_capture_squares():
    """Creates and returns two BoardCell objects corresponding to capture squares.
    """
    w_capture_sq = BoardCell(2, 2)
    b_capture_sq = BoardCell(2, 20)

    return w_capture_sq, b_capture_sq

def uci_move_from_boardcells(source, dest):
    """Returns uci move (as string) from two BoardCells.
    """
    return source.to_chess_sq() + dest.to_chess_sq()

def get_piece_info_from_square(square, grid):
    """Returns tuple of color and piece type from provided square.
    """
    coords = get_chess_coords_from_square(square)
    piece_w_color = grid[coords.row][coords.col]
    if piece_w_color == '.':
        return (None, None)
    color = 'w' if piece_w_color.isupper() else 'b'
    return (color, piece_w_color.lower())

def get_chess_coords_from_square(square):
    """Converts chess square to a BoardCell.

    Example: a1 <=> [0, 0], h8 <=> [7, 7], regardless of whether human plays white or black pieces.
    """
    # Nums correspond to row (rank), letters correspond to col (files)
    return BoardCell(ord(square[1]) - ord('1'), ord(square[0]) - ord('a'))

def get_2d_board(fen, turn=None):
    """Returns a 2d board from fen representation

    Taken from this SO answer:
    https://stackoverflow.com/questions/66451525/how-to-convert-fen-id-onto-a-chess-board
    """
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
            board.append(brow)
        else:
            board.insert(0, brow)
    return board

def is_promotion(prev_board_fen, move):
    """Returns True if move is a promotional move.

    Note: This differs from boardinterface.Engine.is_promotion in that it checks for a promotion
    in the case that the UCI move is not yet known. It is less efficient as it creates a 2d grid
    to check for the piece previously at the square, corresponding to move[:2], and thus should
    only be used when boardinterface.Engine.is_promotion can not be used.
    """
    # If piece in prev_board_fen at square move[:2] is a pawn and move[3] is the final rank,
    # this is a promotion. Note: Don't need to check color since white pawn can't move to row 1
    # and vice versa for black.
    return (get_piece_info_from_square(move[:2], get_2d_board(prev_board_fen))[1] == 'p') and \
           (move[3] in ('1', '8'))

def parse_test_file(fname):
    """Parse provided test file and return a list of moves/messages depending on file type.

    If file type is ".pgn", returns a list of UCI moves. If file type is ".txt", returns a list
    of `Message`. If neither, raises ValueError.
    """
    extension = os.path.splitext(fname)[1]
    if extension not in (".pgn", ".txt"):
        raise ValueError(f"No support for files of type {extension}")

    with open(fname) as file:
        lines = file.readlines()

    messages = []
    if extension == ".pgn":
        # Converts a pgn game string to a list of uci moves; 2nd line contains all moves
        messages = [
            move.uci() for move in chess.pgn.read_game(io.StringIO(lines[1])).mainline_moves()
        ]
    elif extension == ".txt":
        print("Note: When sending debug messages to board or using a test file that is comprised "
              "of `Message` type messages, updating and maintaining game state is not supported.\n"
              "It is required that all messages in a .txt testfile are valid.")
        # Converts a file of `Message` type moves with one message per line to a list
        messages = [line for line in lines if '%' not in line]
        messages = [line.strip('\n') for line in messages if line != '\n']

    return messages, extension

def parse_args():
    """Parse command line arguments.

    Note about priority of modes of operation:
        test > debug > cli == otb == web == speech
    """
    parser = argparse.ArgumentParser(
        description='These allow us to select how we want the game to be played',
        epilog="""The default is to run the chess engine using inputs from player moves detected"""
               """by computer vision.""")

    parser.add_argument('-p', '--playstyle',
                        dest='playstyle',
                        default='cli',
                        help='specifies how human will interact with board during normal play. '
                             'valid options: cli, otb, web, speech')

    parser.add_argument('-d', '--debug',
                        dest='debug',
                        action='store_true',
                        help='if specified, allow sending arbitrary commands to board')

    parser.add_argument('-t', '--test',
                        dest='test',
                        default='',
                        help='if file <TEST> (*.pgn or *.txt) provided, parse program commands '
                             'from specified file')

    parser.add_argument('-m', '--microcontroller',
                        dest='microcontroller',
                        action='store_true',
                        help="if specified, sends commands over UART to Arduino. otherwise, "
                             "simulates communication between pi and Arduino")

    return parser.parse_args()

def transpose_boardcell(boardcell):
    """Transpose board cell from white pieces on human side (default) to black on human side.

    If you transpose a board cell twice, you get the original location.
    Treating this function as `f(x)`, can show that `f^-1(x) == f(x)`. So to go from one
    to the other, can just call this function again (e.g. f(f^-1(x)) == x).
    """
    return BoardCell(Const.BOARD_SIZE - boardcell.row, Const.BOARD_SIZE - boardcell.col)

class Const:
    """Helper file storing various constants used for KG program."""
    CELLS_PER_SQ = 2
    OFFSET_SIZE = 4
    BOARD_SIZE = 22
    UPPER_RIGHT_OFFSET = BOARD_SIZE - OFFSET_SIZE
