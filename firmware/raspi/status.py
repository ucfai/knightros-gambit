from enum import Enum

class ArduinoStatus(Enum):
    # TODO: need to update arduino side codes to match this additional error code
    WaitingInitialization = 0
    WaitingForPersonMove = 1
    WaitingForAIMove = 2
    ExecutingMove = 3
    GenericError = 4

class Status:
    def __init__(self):
        pass

    @staticmethod
    def write_game_status_to_disk(board):
        pass
