'''This module is the interface point between the Arduino and Raspberry Pi.

In addition to encapsulating serial communication to facilitate the game loop,
this file also manages the game state, move validation, etc.
'''
from collections import deque

import chess

from status import ArduinoException, ArduinoStatus, OpCode
import util

class Engine:
    '''Engine designed to be used for maintaining hardware board state.

    Attributes:
        board: A python wrapper around a python-chess board.
    '''
    def __init__(self):
        self.chess_board = chess.Board()

        self.king_to_rook_moves = {}
        self.king_to_rook_moves['e1g1'] = 'h1f1'
        self.king_to_rook_moves['e1c1'] = 'a1d1'
        self.king_to_rook_moves['e8g8'] = 'h8f8'
        self.king_to_rook_moves['e8c8'] = 'a8d8'

    def get_2d_board(self):
        '''Returns a 2d board from fen representation

        Taken from this SO answer:
        https://stackoverflow.com/questions/66451525/how-to-convert-fen-id-onto-a-chess-board
        '''
        fen = self.chess_board.fen()
        board = []
        for row in fen.split('/'):
            brow = []
            for char in row:
                if char == ' ':
                    break

                if char in '12345678':
                    brow.extend(['.'] * int(char))
                else:
                    brow.append(char)
            board.append(brow)
        return board

    def valid_moves_from_position(self):
        '''Returns list of all valid moves (in uci format) from the current board position.
        '''
        return [move.uci() for move in self.chess_board.legal_moves]

    def is_valid_move(self, uci_move):
        '''Returns boolean indicating if move is valid in current board state.
        '''
        try:
            move = self.chess_board.parse_uci(uci_move)
        except ValueError:
            return False

        # Move parsed from UCI can be null, only return True if non-null.
        return bool(move)

    def make_move(self, uci_move):
        '''Updates board state by making move from current position.

        Assumes that provided uci_move is valid.
        '''
        self.chess_board.push_uci(uci_move)

    def fen(self):
        '''Return fen string representation of current board state.
        '''
        return self.chess_board.fen()

    @staticmethod
    def is_promotion(uci_move):
        '''Returns boolean indicating if uci_move is a promotion.
        '''
        if not uci_move or len(uci_move) == 4:
            return False
        return 'q' in uci_move or 'b' in uci_move or 'n' in uci_move or 'r' in uci_move

    def is_castle(self, uci_move):
        '''Returns boolean indicating if uci_move is a castle.
        '''
        return self.chess_board.is_castling(self.chess_board.parse_uci(uci_move))

    def is_capture(self, uci_move):
        '''Returns boolean indicating if uci_move is a capture.
        '''
        return self.chess_board.is_capture(self.chess_board.parse_uci(uci_move))

    @staticmethod
    def get_coords_from_square(square):
        '''Returns tuple of integers indicating grid coordinates from square
        '''
        # TODO: verify that this is the proper indexing for grid created by get_2d_board
        # TODO: make this a BoardCell? Then we need to update the offset when accessing the 2d grid
        return (ord(square[0]) - ord('a'), ord(square[1]) - ord('1'))

    @staticmethod
    def get_coords_from_uci_move(uci_move):
        '''Returns tuple of int tuples, start and end points from provided uci_move.
        '''
        return (Engine.get_coords_from_square(uci_move[:2]),
                Engine.get_coords_from_square(uci_move[2:4]))

    def get_piece_info_from_square(self, square):
        '''Returns tuple of color and piece type from provided square.
        '''
        # TODO: verify this works as intended
        # when given square 'a1' at game start, should return 'w', 'p', etc.
        grid = self.get_2d_board()
        i, j = Engine.get_coords_from_square(square)
        piece_w_color = grid[i][j]
        if piece_w_color != '.':
            return None
        color = 'w' if piece_w_color.isupper() else 'b'
        return (color, piece_w_color.lower())

    def outcome(self):
        '''Returns None if game in progress, otherwise outcome of game.
        '''
        return self.chess_board.outcome()

    def is_game_over(self):
        '''Returns boolean indicating whether or not game is over.
        '''
        return self.chess_board.is_game_over()

class Board:
    '''Main class that serves as glue between software and hardware.

    The Board consists of the interface between the main game loop/AI/computer vision and the
    Arduino code/physical hardware/sensors.

    Attributes:
        engine: See the `Engine` class above: a wrapper around py-chess that offers some
            additional utility functions.
        arduino_status: An enum containing the most recent status received from the Arduino.
        graveyard: See the `Graveyard` class below: A set of coordinates and metadata about
            the dead pieces on the board (captured/spare pieces).
    '''
    def __init__(self):
        self.engine = Engine()
        self.arduino_status = ArduinoStatus.IDLE

        self.graveyard = Graveyard()
        self.move_queue = deque()

    def make_move(self, uci_move):
        ''' This function assumes that is_valid_move has been called for the uci_move.

        Returns true if the uci_move was successfully sent to Arduino
        '''
        try:
            # Send captured piece to graveyard first, then do all the other ops
            if self.engine.is_capture(uci_move):
                self.send_to_graveyard(*self.engine.get_piece_info_from_square(uci_move[2:4]))
            # If move is promotion, send pawn to graveyard, then send promotion piece to board
            if Engine.is_promotion(uci_move):
                # TODO: Add error handling here if the piece we wish to promote to is not
                # available (e.g., all queens have been used already).
                self.handle_promotion(*self.engine.get_piece_info_from_square(uci_move[:2]))
            # If castle, decompose move into king move, then castle move
            if self.engine.is_castle(uci_move):
                # King move
                self.send_message_to_arduino(
                    *Engine.get_coords_from_uci_move(uci_move),
                    opcode=OpCode.MOVE_PIECE_IN_STRAIGHT_LINE)
                # Rook move
                self.send_message_to_arduino(
                    *Engine.get_coords_from_uci_move(self.engine.king_to_rook_moves[uci_move]),
                    opcode=OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
            else:
                if self.is_knight_move_w_neighbors(uci_move):
                    self.send_message_to_arduino(*Engine.get_coords_from_uci_move(uci_move),
                                                 opcode=OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
                else:
                    self.send_message_to_arduino(*Engine.get_coords_from_uci_move(uci_move),
                                                 opcode=OpCode.MOVE_PIECE_IN_STRAIGHT_LINE)
        except ArduinoException as a_e:
            print(f"Unable to send move to Arduino: {a_e.__str__()}")
            return False
        self.engine.make_move(uci_move)
        return True

    def valid_moves_from_position(self):
        '''Returns a list of all valid moves from position.
        '''
        return self.engine.valid_moves_from_position()

    # TODO: implement this function
    def send_message_to_arduino(self, start, end, opcode):
        '''Constructs and sends message according to pi-arduino message format doc.
        '''
        # Have to send metadata about type of move, whether it's a capture/castle/knight move etc
        # This function can also be used for sending moves to graveyard or to promote
        # So need to add move validation as well
        # Maybe message should be constructed before being sent here?
        print("Need to implement sending message to arduino")

    def get_status_from_arduino(self):
        '''Read status from Arduino over UART connection.
        '''
        # TODO: update this function to actually read from arduino
        return self.arduino_status

    def set_status_from_arduino(self, arduino_status=ArduinoStatus.IDLE):
        '''Placeholder function; used for game loop development only.
        '''
        self.arduino_status = arduino_status
        # TODO: send status to Arduino

    def is_valid_move(self, uci_move):
        '''Returns boolean indicating if uci_move is valid in current board state.
        '''
        return self.engine.is_valid_move(uci_move)

    def show_w_graveyard_on_cli(self):
        '''Prints 2d grid of board, also showing which graveyard squares are occupied/empty.
        '''
        print("Need to implement")

    def show_on_cli(self):
        '''Prints board as 2d grid.
        '''
        chess_grid = self.engine.get_2d_board()
        # 8 x 8 chess board
        for i in range(8):
            # Print row, then number indicating rank
            for j in range(8):
                print(chess_grid[i][j], end=' ')
            print(8-i)

        # Print letters under board to indicate files
        for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            print(char, end='')
            print(" ", end='')
        print()

    # def setup_board(self, is_human_turn):
    #     ''' Set up board with black or white on human side, depending on value of is_human_turn.

    #     If is_human_turn == True, white on human side; otherwise, black
    #     '''
    #     print("Need to discuss how board is setup in team meeting")

    def handle_promotion(self, square, color, piece_type):
        '''Assumes that handling captures during promotion done outside this function.

        Backfills promotion area from graveyard if possible.
        '''
        if self.graveyard.dead_piece_counts[color + piece_type] == 0:
            raise ValueError(f"All pieces of type {color}{piece_type} have been used!")

        # First move the pawn being promoted to the graveyard.
        self.send_to_graveyard(color, 'p', origin=square)

        # Put the to-be-promoted piece (from "back" of graveyard) on the appropriate square.
        self.retrieve_from_graveyard(color, piece_type, square)

    def send_to_graveyard(self, color, piece_type, origin=None):
        '''Send piece to graveyard and increment dead piece count.

        Increments the number of dead pieces from k to k + 1, then sends captured piece to
        graveyard[piece types][k + 1]. Note that the piece at index k is the k + 1th piece.
        '''
        self.graveyard.update_dead_piece_count(color, piece_type, delta=1)  # increment
        piece_counts, piece_locs = self.graveyard.get_graveyard_info_for_piece_type(color,
                                                                                    piece_type)
        count = piece_counts[color + piece_type]
        if not origin:
            origin = self.graveyard.w_capture_sq if color == 'w' else self.graveyard.b_capture_sq
        dest = piece_locs[color + piece_type][count]

        self.send_message_to_arduino(origin, dest, OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)

    def retrieve_from_graveyard(self, color, piece_type, destination):
        '''Retrieve piece from the "back" of the graveyard and decrement dead piece count.

        Decrements the number of dead pieces for the retrieved piece type from k to k - 1, then
        sends graveyard piece at graveyard[piece types][k - 1] to `destination`. Note that the
        piece at index k - 1 is the kth piece.
        '''
        piece_counts, piece_locs = self.graveyard.get_graveyard_info_for_piece_type(color,
                                                                                    piece_type)
        count = piece_counts[color + piece_type]
        # Can only retrieve piece from graveyard if there is at least one piece of specified type
        if count == 0:
            raise ValueError(f"There are not enough pieces in the graveyard to support promotion "
                             "to piece of type {piece_type}.")
        self.graveyard.update_dead_piece_count(color, piece_type, delta=-1)  # decrement

        self.send_message_to_arduino(piece_locs[color + piece_type][count-1], destination,
                                     OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)

    def is_knight_move_w_neighbors(self, uci_move):
        '''Return true if uci_move is a knight move with neighbors.
        '''
        # TODO: need to update get_coords_from_uci_move to return board cell
        source, dest = Engine.get_coords_from_uci_move(uci_move)
        _, piece_type = self.engine.get_piece_info_from_square(source)
        if piece_type != "n":
            return False

        board_2d = self.engine.get_2d_board()
        # Cut number of cases from 8 to 4 by treating soure and dest interchangeably
        left, right = (source, dest) if source.col < dest.col else (dest, source)
        if left.col == right.col - 1:
            if left.row == right.row - 2:
                # Case 1, dx=1, dy=2
                return [sq != '.' for sq in [board_2d[left.row][left.col + 1],  # P1
                                             board_2d[left.row + 1][left.col + 1],  # P2
                                             board_2d[left.row + 1][left.col],  # P3
                                             board_2d[left.row + 2][left.col]]]  # P4
            # Else, case 3, dx=1, dy=-2
            return [sq != '.' for sq in [board_2d[left.row][left.col + 1],  # P1
                                         board_2d[left.row - 1][left.col + 1],  # P2
                                         board_2d[left.row - 1][left.col],  # P3
                                         board_2d[left.row - 2][left.col]]]  # P4
        # Else, case 2 or 4
        if left.row == right.row - 1:
            # Case 2, dx=2, dy=1
            return [sq != '.' for sq in [board_2d[left.row + 1][left.col],  # P1
                                         board_2d[left.row + 1][left.col + 1],  # P2
                                         board_2d[left.row][left.col + 1],  # P3
                                         board_2d[left.row][left.col + 2]]]  # P4
        # Else, case 4, dx=2, dy=-1
        return [sq != '.' for sq in [board_2d[left.row - 1][left.col],  # P1
                                     board_2d[left.row - 1][left.col + 1],  # P2
                                     board_2d[left.row][left.col + 1],  # P3
                                     board_2d[left.row][left.col + 2]]]  # P4

    def dispatch_move_from_queue(self):
        if not queue:
            raise ValueError("No moves to dispatch")
        # Note: move is only removed from the queue in the main game loop when the status received
        # from the Arduino confirms that the move has successfully been executed on the Arduino.
        self.send_message_to_arduino(self.move_queue[0])

    def add_move_to_queue(self, source, dest, op_code):
        self.move_count += 1
        move = Move(self.move_count, source, dest, op_code)
        self.move_queue.append(move)

class Graveyard:
    '''Class holds coordinates and state information of board graveyard.

    The graveyard holds captured pieces, and spare pieces used for promotion.

    Attributes:
        dead_piece_counts: A hashmap containing number of dead pieces for each piece type (wp, etc).
        dead_piece_graveyards: A hashmap containing a list of `BoardCell`s for graveyard positions
            of each piece type (wp, etc).
        w_capture_sq: A BoardCell attribute specifying coordinate of white capture square (used by
            human player when they capture a white piece).
        b_capture_sq: A BoardCell attribute specifying coordinate of black capture square (used by
            human player when they capture a black piece).
    '''
    def __init__(self):
        self.dead_piece_counts = util.init_dead_piece_counts()
        self.dead_piece_graveyards = util.init_dead_piece_graveyards()
        self.w_capture_sq, self.b_capture_sq = util.init_capture_squares()

    # TODO: call this method after human makes a move if needed.
    def backfill_promotion_area_from_graveyard(self, color, piece_type):
        '''
        color = 'w' or 'b'
        piece_type in {q, b, n, r}
        Send piece at back of the graveyard to the position to be filled.
        If there are three queens and one is taken, sends Q3 to fill Q1 spot.
        If there are two queens and one is taken, sends Q2 to fill Q1 spot.
        etc...
        '''
        piece_count, graveyard = self.get_graveyard_info_for_piece_type(color, piece_type)

        if piece_count > 1:
            return (graveyard[piece_count-1], graveyard[piece_count-2])
        print("All pieces from graveyard have been used!!")
        return None

    def get_graveyard_info_for_piece_type(self, color, piece_type):
        '''Returns number of dead pieces and list containing coordinates of dead pieces.

        Uses specified piece type and color to index hash map.
        '''
        piece_w_color = color + piece_type
        return self.dead_piece_counts[piece_w_color], self.dead_piece_graveyards[piece_w_color]

    def update_dead_piece_count(self, color, piece_type, delta):
        '''Updates number of dead pieces by specified delta, either 1 or -1.
        '''
        if delta in (1, -1):
            self.dead_piece_counts[color + piece_type] += delta
        else:
            raise ValueError("Cannot modify graveyard in increments greater than 1")

class Move:
    def __init__(self, move_count, source, dest, op_code):
        """Wrapper class for moves from specified source to dest.

        Attributes:
            move_count: int specifying move number in current game. Note, a single chess move may
                be composed of multiple Move objects. For example, a capture is one Move to send
                the captured piece to the graveyard, and another to move the capturing piece to
                the new square.
            source: BoardCell at which to start move.
            dest: BoardCell at which to end move.
            op_code: OpCode specifying how piece should be moved.
        """
        self.move_count = move_count
        self.source = source
        self.dest = dest
        self.op_code = op_code
