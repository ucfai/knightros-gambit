from stockfish import Stockfish

from status import ArduinoStatus, OpCode
from util import BoardCell

class Engine:
    def __init__(self, operating_system, elo_rating=1300):
        self.mac_stockfish_path = "/usr/local/bin/stockfish"
        # TODO: update raspi path
        self.raspi_stockfish_path = "n/a"
        # TODO: update ubuntu path
        self.linux_stockfish_path = "../../chess-engine/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"
        # TODO: update window path
        self.windows_stockfish_path = "n/a"
        operating_system = operating_system.lower()
        if operating_system == "darwin":
            self.stockfish_path = self.mac_stockfish_path
        elif operating_system == "raspi":
            self.stockfish_path = self.raspi_stockfish_path
        elif operating_system == "linux":
            self.stockfish_path = self.linux_stockfish_path
        elif operating_system == "windows":
            self.stockfish_path = self.windows_stockfish_path
        else:
            raise ValueError("Operating system must be one of 'darwin' (osx), 'linux', 'windows', 'raspi'")

        self.stockfish = Stockfish(self.stockfish_path)
        self.stockfish.set_elo_rating(elo_rating)

    def get_2d_board(self):
        '''
        Returns a 2d board from stockfish fen representation
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
                elif char in '12345678':
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
        # 218 is max number of moves from given position:
        # https://www.chessprogramming.org/Chess_Position
        move_dict = self.stockfish.get_top_moves(218)
        return [move['Move'] for move in move_dict]

    def is_valid_move(self, move):
        return self.stockfish.is_move_correct(move)

    def make_move(self, move):
        self.stockfish.make_moves_from_current_position([move])

    def is_promotion(self, move):
        return 'q' in move or 'b' in move or 'n' in move or 'r' in move

    def is_castle(self, move):
        return move == 'e1g1' or move == 'e1c1' or move == 'e8g8' or move == 'e8c8'

class Board:
    def __init__(self, operating_system, elo_rating=1300):
        self.engine = Engine(operating_system, elo_rating)

        self.graveyard = Graveyard()

    def make_move(self, move):
        '''
        This function assumes that is_valid_move has been called for the move
        Returns true if the move was successfully sent to Arduino
        '''
        try:
            if self.engine.is_promotion(move):
                handle_promotion(move)
            if self.engine.is_castle(move):
                # TODO: send the king move as a direct move, then the rook move as indirect
                raise NotImplementedError("Need to create logic for sending castle moves")
            else:
                # TODO: update this to have a real opcode
                self.send_message_to_arduino(move, opcode=None)
        except Exception as e:
            print(f"Unable to send move to Arduino: {e.__str__()}")
            return False
        self.engine.make_move(move)
        return True

    def valid_moves_from_position(self):
        return self.engine.valid_moves_from_position()

    # TODO: implement this function
    def send_message_to_arduino(self, move, opcode):
        # Need to construct message according to pi-arduino message format doc
        # Have to send metadata about type of move, whether it's a capture/castle/knight move etc
        # This function can also be used for sending moves to graveyard or to promote; so need to add move validation as well
        # Maybe message should be constructed before being sent here?
        pass
    
    def get_status_from_arduino(self):
        # TODO: update this function to actually read from arduino. this is just a placeholder for now
        return self.arduino_status

    def set_status_from_arduino(self, arduino_status):
        '''
        Used at startup to set the status of the arduino (i.e. who moves first)
        '''
        self.arduino_status = arduino_status
        # TODO: send status to Arduino

    def is_valid_move(self, move):
        return self.engine.is_valid_move(move)

    def show_on_cli(self):
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
        '''
        Set up board with either black or white on human side, depending on value of is_human_turn.
        If is_human_turn == True, white on human side; otherwise, black
        '''
        # TODO: implement this method
        pass

    def handle_promotion(self, color, piece_type):
        # TODO: implement this method
        pass

class Graveyard:
    def __init__(self):
        # Initialize graveyard positions and values
        self.dead_w_queen_count = 2
        self.w_queen_graveyard = [BoardCell(0, 9), BoardCell(0, 7), BoardCell(1, 7)]
        self.dead_w_bishop_count = 1
        self.w_bishop_graveyard = [BoardCell(1, 9), BoardCell(0, 6), BoardCell(1, 6)]
        self.dead_w_knight_count = 1
        self.w_knight_graveyard = [BoardCell(0, 8), BoardCell(0, 5), BoardCell(1, 5)]
        self.dead_w_rook_count = 1
        self.w_rook_graveyard = [BoardCell(1, 8), BoardCell(0, 4), BoardCell(1, 4)]
        self.dead_w_pawn_count = 0
        self.w_pawn_graveyard = [
            BoardCell(0, 0), BoardCell(1, 0), BoardCell(0, 1), BoardCell(1, 1),
            BoardCell(0, 2), BoardCell(1, 2), BoardCell(0, 3), BoardCell(1, 3), 
        ]
        self.dead_b_queen_count = 2
        self.b_queen_graveyard = [BoardCell(10, 9), BoardCell(10, 7), BoardCell(11, 7)]
        self.dead_b_bishop_count = 1
        self.b_bishop_graveyard = [BoardCell(11, 9), BoardCell(10, 6), BoardCell(11, 6)]
        self.dead_b_knight_count = 1
        self.b_knight_graveyard = [BoardCell(10, 8), BoardCell(10, 5), BoardCell(11, 5)]
        self.dead_b_rook_count = 1
        self.b_rook_graveyard = [BoardCell(11, 8), BoardCell(10, 4), BoardCell(11, 4)]
        self.dead_b_pawn_count = 0
        self.b_pawn_graveyard = [
            BoardCell(10, 0), BoardCell(11, 0), BoardCell(10, 1), BoardCell(11, 1),
            BoardCell(10, 2), BoardCell(11, 2), BoardCell(10, 3), BoardCell(11, 3), 
        ]
        self.w_capture_sq = BoardCell(1, 10)
        self.b_capture_sq = BoardCell(10, 10)


    def backfill_promotion_area_from_graveyard(self, color, piece_type):
        '''
        color = 'w' or 'b'
        piece_type in {q, b, n, r}
        '''
        piece_count, graveyard = self.get_graveyard_info_for_piece_type(color, piece_type)
        i = 1
        while i <= piece_count:
            old_loc = graveyard[i]
            new_loc = graveyard[i-1]
            send_move_to_arduino(f"{OpCode.MovePieceAlongSquareEdges}{old_loc}{new_loc}")

    def get_graveyard_info_for_piece_type(self, color, piece_type):
        if color == 'w':
            if piece_type == 'q':
                return self.dead_w_queen_count, self.w_queen_graveyard
            if piece_type == 'b':
                return self.dead_w_bishop_count, self.w_bishop_graveyard
            if piece_type == 'n':
                return self.dead_w_knight_count, self.w_knight_graveyard
            if piece_type == 'r':
                return self.dead_w_rook_count, self.w_rook_graveyard
            if piece_type == 'p':
                return self.dead_w_pawn_count, self.w_pawn_graveyard
        elif color == 'b':
            if piece_type == 'q':
                return self.dead_b_queen_count, self.b_queen_graveyard
            if piece_type == 'b':
                return self.dead_b_bishop_count, self.b_bishop_graveyard
            if piece_type == 'n':
                return self.dead_b_knight_count, self.b_knight_graveyard
            if piece_type == 'r':
                return self.dead_b_rook_count, self.b_rook_graveyard
            if piece_type == 'p':
                return self.dead_b_pawn_count, self.b_pawn_graveyard
