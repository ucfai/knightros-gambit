from stockfish import Stockfish

from status import ArduinoStatus

class Engine:
    def __init__(self, operating_system, elo_rating=1300):
        self.mac_stockfish_path = "/usr/local/bin/stockfish"
        # TODO: update raspi path
        self.raspi_stockfish_path = "n/a"
        # TODO: update ubuntu path
        self.ubuntu_stockfish_path = "n/a"
        # TODO: update window path
        self.windows_stockfish_path = "n/a"
        operating_system = operating_system.lower()
        if operating_system == "osx":
            self.stockfish_path = self.mac_stockfish_path
        elif operating_system == "raspi":
            self.stockfish_path = self.raspi_stockfish_path
        elif operating_system == "ubuntu":
            self.stockfish_path = self.ubuntu_stockfish_path
        elif operating_system == "windows":
            self.stockfish_path = self.windows_stockfish_path
        else:
            raise ValueError("Operating system must be one of 'osx', 'ubuntu', 'windows', 'raspi'")

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

class Board:
    def __init__(self, operating_system, elo_rating=1300):
        self.engine = Engine(operating_system, elo_rating)
        self.arduino_status = ArduinoStatus.WaitingInitialization

    def make_move(self, move):
        '''
        This function assumes that is_valid_move has been called for the move
        Returns true if the move was successfully sent to Arduino
        '''
        try:
            self.send_move_to_arduino(move)
        except Exception as e:
            print(f"Unable to send move to Arduino: {e.__str__()}")
            return False
        self.engine.make_move(move)
        return True

    def valid_moves_from_position(self):
        return self.engine.valid_moves_from_position()

    # TODO: implement this function
    def send_move_to_arduino(self, move):
        # Need to construct message according to pi-arduino message format doc
        # Have to send metadata about type of move, whether it's a capture/castle/knight move etc
        pass

    # TODO: update this function to actually read from arduino. this is just a placeholder for now
    def get_status_from_arduino(self):
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
