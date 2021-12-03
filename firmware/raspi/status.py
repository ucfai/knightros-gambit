from enum import Enum

class ArduinoStatus(Enum):
    '''Enum of status codes used to indicate current status of Arduino controlling physical board.
    '''
    IDLE = 0
    MESSAGE_IN_PROGRESS = 1
    EXECUTING_MOVE = 2
    ERROR = 3

class Status:
    '''Helper class that stores current status of game, along with related metadata.
    '''
    @staticmethod
    def write_game_status_to_disk(board):
        '''Write pgn to disk, also write associated metadata about board state.
        '''
        raise NotImplementedError("Not implemented")

class OpCode(Enum):
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
    GENERIC_ERROR = 0
