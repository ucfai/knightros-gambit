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
    def __init__(self, human_plays_white_pieces):
        self.chess_board = chess.Board()

        self.human_plays_white_pieces = human_plays_white_pieces

        self.king_to_rook_moves = {}
        self.king_to_rook_moves['e1g1'] = 'h1f1'
        self.king_to_rook_moves['e1c1'] = 'a1d1'
        self.king_to_rook_moves['e8g8'] = 'h8f8'
        self.king_to_rook_moves['e8c8'] = 'a8d8'

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

    def get_board_coords_from_square(self, square):
        '''Returns tuple of integers indicating grid coordinates from square
        '''
        sq_to_xy = util.get_chess_coords_from_square(square)

        # From top down perspective with human on "bottom", bottom left corner is (0, 0)
        # Each cell (which is the size of one chess square) is split into two unit spaces. This
        # allows accessing the corners and edges of squares. This scheme is needed in order to
        # implement the cache_captured_piece function.
        # If human plays white pieces, then the middle of square "a1" corresponds to BoardCell
        # (5, 5) and the middle of "h8" corresponds to BoardCell (19, 19). Vice versa for black
        # pieces. Below logic converts chess coordinates to board coordinates.
        if self.human_plays_white_pieces:
            return util.BoardCell((sq_to_xy.row * 2) + 5, (sq_to_xy.col * 2) + 5)

        return util.BoardCell(19 - (sq_to_xy.row * 2), 19 - (sq_to_xy.col * 2))

    @staticmethod
    def get_chess_coords_from_uci_move(uci_move):
        '''Returns tuple of BoardCells w.r.t. 8x8 chess grid.

        BoardCells specifying start and end points from provided uci_move.
        '''
        # TODO: consider case of promotion. Maybe this should return three-tuple with optional
        # third value that is None when not a promotion.
        return (util.get_chess_coords_from_square(uci_move[:2]),
                util.get_chess_coords_from_square(uci_move[2:4]))

    def get_board_coords_from_uci_move(self, uci_move):
        '''Returns tuple of BoardCells w.r.t. physical board (including graveyard and edges).

        BoardCells specifying start and end points from provided uci_move.
        '''
        # TODO: consider case of promotion. Maybe this should return three-tuple with optional
        # third value that is None when not a promotion.
        return (self.get_board_coords_from_square(uci_move[:2]),
                self.get_board_coords_from_square(uci_move[2:4]))

    def get_piece_info_from_square(self, square):
        '''Returns tuple of color and piece type from provided square.
        '''
        return util.get_piece_info_from_square(square, util.get_2d_board(self.fen()))

    def outcome(self):
        '''Returns None if game in progress, otherwise outcome of game.
        '''
        return self.chess_board.outcome()

    def is_game_over(self):
        '''Returns boolean indicating whether or not game is over.
        '''
        return self.chess_board.is_game_over()

    def get_safe_corner(self, uci_move):
        '''Returns a "safe" corner on which to cache a captured piece before sending to graveyard.

        A "safe" corner is one that is not in the way of the capturing piece's path to the capture
        square. This function determines unsafe corner(s) (adjacent to the moving piece) and
        returns one of the other corners.
        '''
        source, dest = self.get_board_coords_from_uci_move(uci_move)
        if source.row >= dest.row:  # Can't use top two corners
            if source.col >= dest.col:  # Can't use bottom right corner
                # Use bottom left corner of dest to cache captured piece
                return util.BoardCell(dest.row - 1, dest.col - 1)
            # Use bottom right corner of dest to cache captured piece
            return util.BoardCell(dest.row - 1, dest.col + 1)
        # Source row < dest row, so we can't use bottom two corners
        if source.col >= dest.col:  # Can't use top right corner
            # Use top left corner of dest to cache captured piece
            return util.BoardCell(dest.row + 1, dest.col - 1)
        # Use top right corner of dest to cache captured piece
        return util.BoardCell(dest.row + 1, dest.col + 1)

    def is_en_passant(self, uci_move):
        '''Return true if the given uci_move is an en passant.
        '''
        return self.chess_board.is_en_passant(self.chess_board.parse_uci(uci_move))

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
    def __init__(self, human_plays_white_pieces):
        self.engine = Engine(human_plays_white_pieces)
        self.move_count = 0
        self.arduino_status = ArduinoStatus(ArduinoStatus.IDLE, self.move_count, None)

        self.graveyard = Graveyard()
        self.move_queue = deque()
        self.human_plays_white_pieces = human_plays_white_pieces

    def send_move_to_board(self, uci_move):
        '''Validate move and send to board interface.
        '''
        if self.is_valid_move(uci_move):
            self.make_move(uci_move)
        else:
            # TODO: do error handling
            raise NotImplementedError("Need to handle case of invalid move input. "
                                      "Should we loop until move is valid? What if "
                                      "the board is messed up? Need to revisit.")

    def make_move(self, uci_move):
        ''' This function assumes that is_valid_move has been called for the uci_move.

        Returns true if the uci_move was successfully sent to Arduino
        '''
        try:
            # `cache_info` is used at end of the move sequence to move the captured piece from the
            # cached location (a safe corner not in the way of the capturing piece) to the
            # graveyard. This is done rather than moving the captured piece to the graveyard
            # before moving the capturing piece in order to minimize the total amount of actuation.
            cache_info = self.cache_captured_piece(uci_move)
            # Handle promotion moves that are not captures
            if Engine.is_promotion(uci_move):
                # TODO: Add error handling here if the piece we wish to promote to is not
                # available (e.g., all queens have been used already).
                self.handle_promotion(uci_move)
            # If castle, decompose move into king move, then rook move
            elif self.engine.is_castle(uci_move):
                # King move
                self.add_move_to_queue(
                    *self.engine.get_board_coords_from_uci_move(uci_move),
                    OpCode.MOVE_PIECE_IN_STRAIGHT_LINE)
                # Rook move
                self.add_move_to_queue(
                    *self.engine.get_board_coords_from_uci_move(
                        self.engine.king_to_rook_moves[uci_move]),
                    OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
            else:
                if self.is_knight_move_w_neighbors(uci_move):
                    self.add_move_to_queue(*self.engine.get_board_coords_from_uci_move(uci_move),
                                           OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
                else:
                    self.add_move_to_queue(*self.engine.get_board_coords_from_uci_move(uci_move),
                                           OpCode.MOVE_PIECE_IN_STRAIGHT_LINE)
            if cache_info:
                self.send_to_graveyard(*cache_info)
        except ArduinoException as a_e:
            print(f"Unable to send move to Arduino: {a_e.__str__()}")
            return False
        self.engine.make_move(uci_move)
        return True

    def valid_moves_from_position(self):
        '''Returns a list of all valid moves from position.
        '''
        return self.engine.valid_moves_from_position()

    def send_message_to_arduino(self, board_move):
        '''Constructs and sends message according to pi-arduino message format doc.
        '''
        # Convert BoardCell integer coordinates to chars s.t. 0 <=> 'A', ..., 11 <=> 'L' before
        # sending message so that each coord only takes up one byte. This will need to be
        # converted back on the Arduino side.
        source_str = chr(board_move.source.row + ord('A')) + chr(board_move.source.col + ord('A'))
        dest_str = chr(board_move.dest.row + ord('A')) + chr(board_move.dest.col + ord('A'))
        msg = f"~{board_move.op_code}{source_str}{dest_str}{board_move.move_count % 10}"

        print(f"Sending message \"{msg}\" to arduino")

        # TODO: Implement sending message to arduino

        # TODO: This is for game loop dev, remove once we read from arduino
        self.set_status_from_arduino(ArduinoStatus.EXECUTING_MOVE, board_move.move_count, None)

    def get_status_from_arduino(self):
        '''Read status from Arduino over UART connection.
        '''
        # TODO: update this function to actually read from arduino
        return self.arduino_status

    def set_status_from_arduino(self, status, move_count, extra):
        '''Placeholder function; used for game loop development only.
        '''
        self.arduino_status = ArduinoStatus(status, move_count, extra)
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
        # (0, 0) corresponds to a1, want to print s.t. a1 is bottom left, so reverse rows
        chess_grid = util.get_2d_board(self.engine.fen())
        chess_grid.reverse()
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

    def handle_promotion(self, uci_move):
        '''Assumes that handling captures during promotion done outside this function.

        Sends pawn to graveyard, then sends promotion piece to board.
        '''
        square = self.engine.get_board_coords_from_square(uci_move[2:4])
        color = self.engine.get_piece_info_from_square(uci_move[2:4])[0]
        piece_type = uci_move[4]

        if self.graveyard.dead_piece_counts[color + piece_type] == 0:
            raise ValueError(f"All pieces of type {color}{piece_type} have been used!")

        # First move the pawn being promoted to the graveyard.
        self.send_to_graveyard(color, 'p', origin=square)

        # Put the to-be-promoted piece (from "back" of graveyard) on the appropriate square.
        self.retrieve_from_graveyard(color, piece_type, square)

    def cache_captured_piece(self, uci_move):
        '''If uci_move is a capture, cache the captured piece on a safe space.

        If move is an en passant, the cache location is simply the original square. If it is any
        other type of capture, the cache location is one of the corners of the destination square
        that is not in the way of the capturing piece.

        If uci_move is not a capture, return value will be None. Else, return value is a tuple of
        `(piece_color, piece_type, cache_loc)`.

        Returns a tuple of  location of cached piece as a BoardCell, if applicable.
        '''
        if not self.engine.is_capture(uci_move):
            return None

        # Note: en passant square given by dest.col (uci_move[2]) + source.row (uci_move[1])
        if self.engine.is_en_passant(uci_move):
            square = uci_move[2] + uci_move[1]
            cache_loc = self.engine.get_board_coords_from_square(uci_move[2] + uci_move[1])
        else:
            square = uci_move[2:4]
            cache_loc = self.engine.get_safe_corner(uci_move)
            # Move the piece at uci_move[2:4] to cache_loc
            self.add_move_to_queue(self.engine.get_board_coords_from_square(uci_move[2:4]),
                                   cache_loc, OpCode.MOVE_PIECE_IN_STRAIGHT_LINE)

        # `get_piece_info_from_square()` returns a tuple, + operator concatenates before returning
        return self.engine.get_piece_info_from_square(square) + (cache_loc,)

    def send_to_graveyard(self, color, piece_type, origin=None):
        '''Send piece to graveyard and increment dead piece count.

        Increments the number of dead pieces from k to k + 1, then sends captured piece to
        graveyard[piece types][k + 1]. Note that the piece at index k is the k + 1th piece.
        '''
        self.graveyard.update_dead_piece_count(color, piece_type, delta=1)  # increment
        count, piece_locs = self.graveyard.get_graveyard_info_for_piece_type(color, piece_type)

        if not origin:
            origin = self.graveyard.w_capture_sq if color == 'w' else self.graveyard.b_capture_sq
        dest = piece_locs[count]

        self.add_move_to_queue(origin, dest, OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)

    def retrieve_from_graveyard(self, color, piece_type, destination):
        '''Retrieve piece from the "back" of the graveyard and decrement dead piece count.

        Decrements the number of dead pieces for the retrieved piece type from k to k - 1, then
        sends graveyard piece at graveyard[piece types][k - 1] to `destination`. Note that the
        piece at index k - 1 is the kth piece.
        '''
        count, piece_locs = self.graveyard.get_graveyard_info_for_piece_type(color, piece_type)

        # Can only retrieve piece from graveyard if there is at least one piece of specified type
        if count == 0:
            raise ValueError(f"There are not enough pieces in the graveyard to support promotion "
                             "to piece of type {piece_type}.")
        self.graveyard.update_dead_piece_count(color, piece_type, delta=-1)  # decrement

        self.add_move_to_queue(piece_locs[count-1], destination,
                               OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)

    def is_knight_move_w_neighbors(self, uci_move):
        '''Return true if uci_move is a knight move with neighbors.
        '''
        _, piece_type = self.engine.get_piece_info_from_square(uci_move[:2])
        if piece_type != "n":
            return False

        source, dest = Engine.get_chess_coords_from_uci_move(uci_move)

        board_2d = util.get_2d_board(self.engine.fen())
        # Cut number of cases from 8 to 4 by treating soure and dest interchangeably
        left, right = (source, dest) if source.col < dest.col else (dest, source)
        if left.col == right.col - 1:
            if left.row == right.row - 2:
                # Case 1, dx=1, dy=2
                return any([sq != '.' for sq in [board_2d[left.row][left.col + 1],  # P1
                                                 board_2d[left.row + 1][left.col + 1],  # P2
                                                 board_2d[left.row + 1][left.col],  # P3
                                                 board_2d[left.row + 2][left.col]]])  # P4
            # Else, case 3, dx=1, dy=-2
            return any([sq != '.' for sq in [board_2d[left.row][left.col + 1],  # P1
                                             board_2d[left.row - 1][left.col + 1],  # P2
                                             board_2d[left.row - 1][left.col],  # P3
                                             board_2d[left.row - 2][left.col]]])  # P4
        # Else, case 2 or 4
        if left.row == right.row - 1:
            # Case 2, dx=2, dy=1
            return any([sq != '.' for sq in [board_2d[left.row + 1][left.col],  # P1
                                             board_2d[left.row + 1][left.col + 1],  # P2
                                             board_2d[left.row][left.col + 1],  # P3
                                             board_2d[left.row][left.col + 2]]])  # P4
        # Else, case 4, dx=2, dy=-1
        return any([sq != '.' for sq in [board_2d[left.row - 1][left.col],  # P1
                                         board_2d[left.row - 1][left.col + 1],  # P2
                                         board_2d[left.row][left.col + 1],  # P3
                                         board_2d[left.row][left.col + 2]]])  # P4

    def dispatch_move_from_queue(self):
        '''Send move at front of self.move_queue to Arduino, if queue is non-empty.
        '''
        if not self.move_queue:
            raise ValueError("No moves to dispatch")
        # Note: move is only removed from the queue in the main game loop when the status received
        # from the Arduino confirms that the move has successfully been executed on the Arduino.
        self.send_message_to_arduino(self.move_queue[0])

    def add_move_to_queue(self, source, dest, op_code):
        '''Increment move count and add Move to self.move_queue.
        '''
        self.move_count += 1
        move = Move(self.move_count, source, dest, op_code)
        self.move_queue.append(move)

    def backfill_promotion_area_from_graveyard(self, color, piece_type):
        '''
        color = 'w' or 'b'
        piece_type in {q, b, n, r}
        Send piece at back of the graveyard to the position to be filled.
        If there are three queens and one is taken, sends Q3 to fill Q1 spot.
        If there are two queens and one is taken, sends Q2 to fill Q1 spot.
        etc...
        '''
        count, graveyard = self.graveyard.get_graveyard_info_for_piece_type(color, piece_type)

        # This method is only called if human promoted a piece, which means they took a piece
        # from the promotion area of the graveyard. Thus, it's always safe to decrement dead
        # piece count.
        self.graveyard.update_dead_piece_count(color, piece_type, delta=-1)  # decrement
        if count > 0:
            self.add_move_to_queue(graveyard[count-1], graveyard[0],
                                   OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
        raise ValueError("All promotional pieces from graveyard have been used!!")


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
    def __init__(self, move_count, source, dest, op_code):
        self.move_count = move_count
        self.source = source
        self.dest = dest
        self.op_code = op_code
