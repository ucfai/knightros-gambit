'''This module is the interface point between the Arduino and Raspberry Pi.

In addition to encapsulating serial communication to facilitate the game loop,
this file also manages the game state, move validation, etc.
'''

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

    @staticmethod
    def get_coords_from_square(square):
        '''Returns tuple of integers indicating grid coordinates from square
        '''
        # TODO: verify that this is the proper indexing for grid created by get_2d_grid
        # TODO: make this a BoardCell? Then we need to update the offset when accessing the 2d grid
        return (ord(square[0]) - ord('a'), ord(square[1]) - ord('1'))

    @staticmethod
    def get_coords_from_uci_move(uci_move):
        '''Returns tuple of int tuples, start and end points from provided uci_move.
        '''
        return (Engine.get_coords_from_square(uci_move[:2]),
                Engine.get_coords_from_square(uci_move[2:]))

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

    def make_move(self, uci_move):
        ''' This function assumes that is_valid_move has been called for the uci_move.

        Returns true if the uci_move was successfully sent to Arduino
        '''
        try:
            if Engine.is_promotion(uci_move):
                self.handle_promotion(*self.engine.get_piece_info_from_square(uci_move[:2]))
            if self.engine.is_castle(uci_move):
                # TODO: send the king move as a direct uci_move, then the rook move as indirect
                print("Need to create logic for sending castle moves")
            else:
                # TODO: update this to have a real opcode
                self.send_message_to_arduino(*Engine.get_coords_from_uci_move(uci_move),
                                             opcode=None)
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
        pass

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
        pass

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

    def handle_promotion(self, color, piece_type):
        '''Backfills promotion area from graveyard if possible.
        '''
        if self.graveyard.dead_piece_counts[color + piece_type] > 1:
            self.send_message_to_arduino(
                *self.graveyard.backfill_promotion_area_from_graveyard(color, piece_type),
                OpCode.MOVE_PIECE_ALONG_SQUARE_EDGES)
            self.graveyard.update_dead_piece_count(color, piece_type, delta=1)
        else:
            print(f"All pieces of type {color}{piece_type} have been used!")

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
