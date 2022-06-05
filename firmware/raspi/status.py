"""Helper class storing status messages used in Arduino/Raspi communication.
"""

class ArduinoStatus:
    """Wrapper around messages received from Arduino.
    # TODO: add in some more documentation here w.r.t. message formatting.
    """
    # Status codes used to indicate current status of Arduino controlling physical board.
    IDLE = '0'
    EXECUTING_MOVE = '1'
    END_TURN_BUTTON_PRESSED = '2'
    ERROR = '3'

    # Arduino Error Codes
    INVALID_OP = '0'
    INVALID_LOCATION = '1'
    MOVEMENT_ERROR = '2'

    # Incoming message length from Arduino
    MESSAGE_LENGTH = 4

    # TODO: Update order of fields so that it matches actual msg format. Should be status, extra,
    # move_count, in that order.
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
        if self.status == ArduinoStatus.EXECUTING_MOVE:
            return "EXECUTING_MOVE"
        if self.status == ArduinoStatus.END_TURN_BUTTON_PRESSED:
            return "END_TURN_BUTTON_PRESSED"
        if self.status == ArduinoStatus.ERROR:
            return "ERROR"
        return ""

    @staticmethod
    def is_valid_code(code):
        """Returns True if provided code is a valid ArduinoStatus code, else False.
        """
        return code in (ArduinoStatus.IDLE,
                        ArduinoStatus.EXECUTING_MOVE,
                        ArduinoStatus.END_TURN_BUTTON_PRESSED,
                        ArduinoStatus.ERROR)

class Status:
    """Helper class that stores current status of game, along with related metadata.
    """
    @staticmethod
    def write_game_status_to_disk(board):
        """Write pgn to disk, also write associated metadata about board state.
        """
        raise NotImplementedError("Not implemented")

class OpCode:
    """Enum of op codes used to provide information about type of move Arduino should make.
    """
    # This code indicates Arduino should use straight-line path, diagonals allowed
    MOVE_PIECE_IN_STRAIGHT_LINE = '0'

    # This code indicates Arduino should move piece along square edges instead of
    # through the center of squares. Used for knights if adjacent pieces, for
    # graveyard pathing, and for castling
    MOVE_PIECE_ALONG_SQUARE_EDGES = '1'

    # Queenside and kingside castling are composed of two moves; one `MOVE_PIECE_IN_STRAIGHT_LINE`
    # (king moving in straight line) and one `MOVE_PIECE_ALONG_SQUARE_EDGES`

    # This code indicates Arduino should use electromagnet to center piece on square
    ALIGN_PIECE_ON_SQUARE = '2'

    # OpCodes that use a UCI move to generate a movement type message
    UCI_MOVE_OPCODES = (
        MOVE_PIECE_IN_STRAIGHT_LINE,
        MOVE_PIECE_ALONG_SQUARE_EDGES,
        ALIGN_PIECE_ON_SQUARE)

    # OpCode that specifies Arduino should perform a non-UCI move action
    INSTRUCTION = '3'

    MESSAGE_LENGTH = 7

class InstructionType:
    """Enum of instruction types used to provide information about OpCode.INSTRUCTION types."""
    # This code indicates Arduino should align an axis
    # Setting Extra byte (e.g. msg[2]) to '0' indicates aligning x axis to zero, '1' indicates
    # aligning y axis to zero, '2' indicates aligning x axis to max, '3' indicates aligning y axis
    # to max, '4' indicates calling the Arduino's home() function, which aligns x to zero and y to
    # zero.
    ALIGN_AXIS = 'A'

    # This code indicates Arduino should set the state of the electromagnet
    # Setting Extra byte (e.g. msg[2]) to '0' indicates OFF, '1' indicates ON
    SET_ELECTROMAGNET = 'S'

    # A code used to set the human_move_valid_flag, which guards button presses. When the Extra byte
    # is '1', allows button presses; when '0', disallows button presses (for the chess timer, which
    # signals end of human turn).
    SET_HUMAN_MOVE_VALID = 'M'

    # This code indicates Arduino should retransmit last message
    # This code used when a corrupted or misaligned message is received
    RETRANSMIT_LAST_MSG = 'R'

    # A code used to indicate that the Arduino should enable/disable the motors
    # Setting the Extra field (e.g. msg[2] to '0' indicates disabling the motors, '1' indicates
    # enabling the motors.
    ENABLE_MOTORS = 'E'

    # A code used to indicate the Arduino should restart (perform a power cycle)
    RESTART_ARDUINO = 'P'

    # Tuple of all InstructionTypes, used for checking membership
    VALID_INSTRUCTIONS = (
        ALIGN_AXIS,             # A
        SET_ELECTROMAGNET,      # S
        SET_HUMAN_MOVE_VALID,   # M
        RETRANSMIT_LAST_MSG,    # R
        ENABLE_MOTORS,          # E
        RESTART_ARDUINO,          # P
    )

class ArduinoException(Exception):
    """Helper class for custom Arduino exceptions.
    """
    NO_ERROR = '0'
    INVALID_OP = '1'
    INVALID_LOCATION = '2'
    INCOMPLETE_INSTRUCTION = '3'
    MOVEMENT_ERROR = '4'
