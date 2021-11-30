from stockfish import Stockfish

from status import ArduinoException, ArduinoStatus, OpCode
from util import BoardCell

class Engine:
    '''Wrapper around the Stockfish engine.

    Provides some additional utility functions not offered by Stockfish.

    Attributes:
        stockfish: A python wrapper around the Stockfish chess engine.
    '''
    def __init__(self, operating_system):
        operating_system = operating_system.lower()
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

        self.stockfish = Stockfish(stockfish_path)

    def get_2d_board(self):
        '''Returns a 2d board from stockfish fen representation
        
        Taken from this SO answer:
        https://stackoverflow.com/questions/66451525/how-to-convert-fen-id-onto-a-chess-board
        '''
        fen = self.stockfish.get_fen_position()
        board = []
        for row in fen.split('/'):
            brow = []
            for char in row:
                if char == ' ':
                    break

                if char in '12345678':
                    brow.extend(['--'] * int(char))
                elif char == 'p':
                    brow.append('bp')
                elif char == 'P':
                    brow.append('wp')
                elif char > 'Z':
                    brow.append('b'+char.upper())
                else:
                    brow.append('w'+char)
            board.append(brow)
        return board

    def valid_moves_from_position(self):
        '''Returns all valid moves from the current board position.

        218 is max number of moves from given position:
        https://www.chessprogramming.org/Chess_Position
        '''
        move_dict = self.stockfish.get_top_moves(218)
        return [move['Move'] for move in move_dict]

    def is_valid_move(self, move):
        '''Returns boolean indicating if move is valid in current board state.
        '''
        return self.stockfish.is_move_correct(move)

    def make_move(self, move):
        '''Updates board state by making move from current position.

        Assumes that provided move is valid.
        '''
        self.stockfish.make_moves_from_current_position([move])

    def get_fen_position(self):
        return self.stockfish.get_fen_position()

    @staticmethod
    def is_promotion(move):
        '''Returns boolean indicating if move is a promotion.
        '''
        if not move:
            return False
        return 'q' in move or 'b' in move or 'n' in move or 'r' in move

    @staticmethod
    def is_castle(move):
        '''Returns boolean indicating if move is a castle.
        '''
        return move in ('e1g1', 'e1c1', 'e8g8', 'e8c8')

    @staticmethod
    def get_coords_from_square(square):
        '''Returns tuple of integers indicating grid coordinates from square
        '''
        # TODO: verify that this is the proper indexing for grid created by get_2d_grid
        # TODO: make this a BoardCell? Then we need to update the offset when accessing the 2d grid
        return (ord(square[0]) - ord('a'), ord(square[1]) - ord('1'))

    @staticmethod
    def get_start_and_end_coords_from_move(move):
        '''Returns tuple of int tuples, start and end points from provided move.
        '''
        return (Engine.get_coords_from_square(move[:2]), Engine.get_coords_from_square(move[2:]))

    def get_piece_info_from_square(self, square):
        '''Returns tuple of color and piece type from provided square.
        '''
        # TODO: verify this works as intended
        # when given square 'a1' at game start, should return 'w', 'p', etc.
        grid = self.get_2d_board()
        i, j = Engine.get_coords_from_square(square)
        piece_w_color = grid[i][j]
        if piece_w_color != "--":
            return None
        return (piece_w_color[0], piece_w_color[1].lower())

class Board:
    '''Main class that serves as glue between software and hardware.

    The Board consists of the interface between the main game loop/AI/computer vision and the
    Arduino code/physical hardware/sensors.

    Attributes:
        engine: See the `Engine` class above: a wrapper around Stockfish that offers some
            additional utility functions.
        arduino_status: An enum containing the most recent status received from the Arduino.
        graveyard: See the `Graveyard` class below: A set of coordinates and metadata about
            the dead pieces on the board (captured/spare pieces).
    '''
    def __init__(self, operating_system):
        self.engine = Engine(operating_system)
        self.arduino_status = ArduinoStatus.Idle

        self.graveyard = Graveyard()

    def make_move(self, move):
        ''' This function assumes that is_valid_move has been called for the move.

        Returns true if the move was successfully sent to Arduino
        '''
        try:
            if Engine.is_promotion(move):
                self.handle_promotion(*self.engine.get_piece_info_from_square(move[:2]))
            if Engine.is_castle(move):
                # TODO: send the king move as a direct move, then the rook move as indirect
                print("Need to create logic for sending castle moves")
            else:
                # TODO: update this to have a real opcode
                self.send_message_to_arduino(*Engine.get_start_and_end_coords_from_move(move),
                                             opcode=None)
        except ArduinoException as a_e:
            print(f"Unable to send move to Arduino: {a_e.__str__()}")
            return False
        self.engine.make_move(move)
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

    def set_status_from_arduino(self, arduino_status=ArduinoStatus.Idle):
        '''Placeholder function; used for game loop development only.
        '''
        self.arduino_status = arduino_status
        # TODO: send status to Arduino

    def is_valid_move(self, move):
        '''Returns boolean indicating if move is valid in current board state.
        '''
        return self.engine.is_valid_move(move)

    def show_on_cli(self):
        '''Prints board as 2d grid.
        '''
        boardstate = self.engine.stockfish.get_board_visual()
        # Last row is newline
        for brow in boardstate.split('\n')[:-1]:
            print(brow)
        # Print letters under grid to show files
        print("  ", end='')
        for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            print(char, end='')
            print("   ", end='')
        print()

    def setup_board(self, is_human_turn):
        ''' Set up board with either black or white on human side, depending on value of is_human_turn.

        If is_human_turn == True, white on human side; otherwise, black
        '''
        # TODO: implement this method
        print("Need to discuss how board is setup in team meeting")

    def handle_promotion(self, color, piece_type):
        '''Backfills promotion area from graveyard if possible.
        '''
        if graveyard.dead_piece_counts[color + piece_type] > 1:
            self.send_message_to_arduino(
                *self.graveyard.backfill_promotion_area_from_graveyard(color, piece_type),
                OpCode.MovePieceAlongSquareEdges)
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
        self.dead_piece_counts = {}
        self.dead_piece_graveyards = {}

        self.dead_piece_counts["wq"] = 2
        self.dead_piece_graveyards["wq"] = [BoardCell(0, 9), BoardCell(0, 7), BoardCell(1, 7)]
        self.dead_piece_counts["wb"] = 1
        self.dead_piece_graveyards["wb"] = [BoardCell(1, 9), BoardCell(0, 6), BoardCell(1, 6)]
        self.dead_piece_counts["wn"] = 1
        self.dead_piece_graveyards["wn"] = [BoardCell(0, 8), BoardCell(0, 5), BoardCell(1, 5)]
        self.dead_piece_counts["wr"] = 1
        self.dead_piece_graveyards["wr"] = [BoardCell(1, 8), BoardCell(0, 4), BoardCell(1, 4)]
        self.dead_piece_counts["wp"] = 0
        self.dead_piece_graveyards["wp"] = [
            BoardCell(0, 0), BoardCell(1, 0), BoardCell(0, 1), BoardCell(1, 1),
            BoardCell(0, 2), BoardCell(1, 2), BoardCell(0, 3), BoardCell(1, 3),
        ]

        self.dead_piece_counts["bq"] = 2
        self.dead_piece_graveyards["bq"] = [BoardCell(10, 9), BoardCell(10, 7), BoardCell(11, 7)]
        self.dead_piece_counts["bb"] = 1
        self.dead_piece_graveyards["bb"] = [BoardCell(11, 9), BoardCell(10, 6), BoardCell(11, 6)]
        self.dead_piece_counts["bn"] = 1
        self.dead_piece_graveyards["bn"] = [BoardCell(10, 8), BoardCell(10, 5), BoardCell(11, 5)]
        self.dead_piece_counts["br"] = 1
        self.dead_piece_graveyards["br"] = [BoardCell(11, 8), BoardCell(10, 4), BoardCell(11, 4)]
        self.dead_piece_counts["bp"] = 0
        self.dead_piece_graveyards["bp"] = [
            BoardCell(10, 0), BoardCell(11, 0), BoardCell(10, 1), BoardCell(11, 1),
            BoardCell(10, 2), BoardCell(11, 2), BoardCell(10, 3), BoardCell(11, 3),
        ]

        self.w_capture_sq = BoardCell(1, 10)
        self.b_capture_sq = BoardCell(10, 10)


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
        if delta == 1 or delta == -1:
            self.dead_piece_counts[color + piece_type] += delta
        else:
            raise ValueError("Cannot modify graveyard in increments greater than 1")
