from enum import Enum

class ArduinoStatus(Enum):
    # TODO: need to update arduino side codes to match this additional error code
    Idle = 0
    MessageInProgress = 1
    ExecutingMove = 2
    Error = 3

class Status:
    def __init__(self):
        pass

    @staticmethod
    def write_game_status_to_disk(board):
        pass

class OpCode(Enum):
    # This code indicates Arduino should use straight-line path, diagonals allowed
    MovePieceInStraightLine = 0
    # This code used for knights if adjacent pieces, for graveyard pathing, and for castling
    MovePieceAlongSquareEdges = 1
    # Queenside and kingside castling are composed of two moves; one `SimpleMove` (king moving in straight line)
    # and one `MovePieceAlongSquareEdges`
