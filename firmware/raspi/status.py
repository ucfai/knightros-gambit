class ArduinoStatus:
    '''Wrapper around messages received from Arduino.
    # TODO: add in some more documentation here w.r.t. message formatting.
    '''
    # Status codes used to indicate current status of Arduino controlling physical board.
    IDLE = 0
    MESSAGE_IN_PROGRESS = 1
    EXECUTING_MOVE = 2
    END_TURN_BUTTON_PRESSED = 3
    ERROR = 4

    def __init__(self, status, move_count, extra):
        self.status = status
        self.move_count = move_count
        self.extra = extra

    @staticmethod
    def parse_message(message):
        """Parse message to construct and return an ArduinoStatus.
        """
        if message[0] != '~' or len(message) != 4:
            return None
        return ArduinoStatus(message[1], message[2], message[3])

    def __str__(self):
        if self.status == ArduinoStatus.IDLE:
            return "IDLE"
        if self.status == ArduinoStatus.MESSAGE_IN_PROGRESS:
            return "MESSAGE_IN_PROGRESS"
        if self.status == ArduinoStatus.EXECUTING_MOVE:
            return "EXECUTING_MOVE"
        if self.status == ArduinoStatus.END_TURN_BUTTON_PRESSED:
            return "END_TURN_BUTTON_PRESSED"
        if self.status == ArduinoStatus.ERROR:
            return "ERROR"
        return ""

class Status:
    '''Helper class that stores current status of game, along with related metadata.
    '''
    @staticmethod
    def write_game_status_to_disk(board):
        '''Write pgn to disk, also write associated metadata about board state.
        '''
        raise NotImplementedError("Not implemented")

class OpCode:
    '''Enum of op codes used to provide information about type of move Arduino should make.
    '''
    # This code indicates Arduino should use straight-line path, diagonals allowed
    MOVE_PIECE_IN_STRAIGHT_LINE = 0

    # This code indicates Arduino should move piece along square edges instead of
    # through the center of squares. Used for knights if adjacent pieces, for
    # graveyard pathing, and for castling
    MOVE_PIECE_ALONG_SQUARE_EDGES = 1

    # Queenside and kingside castling are composed of two moves; one `MOVE_PIECE_IN_STRAIGHT_LINE`
    # (king moving in straight line) and one `MOVE_PIECE_ALONG_SQUARE_EDGES`

    # This code indicates Arduino should use electromagnet to center piece on square
    ALIGN_PIECE_ON_SQUARE = 2

class ArduinoException(Exception):
    '''Helper class for custom Arduino exceptions.
    '''
    NONE = 0
    INVALID_OP = 1
    INVALID_LOCATION = 2
