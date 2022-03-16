'''Main game class that serves as backend for controlling automatic chessboard.

Can be used with multiple different front end options, `cliinterface.py`, many others still being
developed (web, speech, otb).
'''
import time

from boardinterface import Board
from status import ArduinoStatus, OpCode
import util

class Game:
    """Main game class. All frontends should have an instance of this class.

    The main program logic is in the `process` function. `process` should be called in a loop in
    the frontend code.
    """
    def __init__(self, mode_of_interaction, human_plays_white_pieces=None):
        self.mode_of_interaction = mode_of_interaction
        # TODO: Set up board with either white or black on human side.
        self.board = Board(human_plays_white_pieces)
        # board.setup_board(is_human_turn)

        # TODO: remove this after real Arduino communication is set up
        self.board.set_status_from_arduino(ArduinoStatus.IDLE, 0, None)

    def winner(self):
        """Returns None if draw or still in progress, True if white won, False if black won.
        """
        return self.board.engine.outcome()

    def current_fen(self):
        """Returns current board fen string.
        """
        return self.board.engine.fen()

    def is_white_turn(self):
        '''Return True if it is white's turn, False otherwise.
        '''
        return self.board.engine.is_white_turn()

    def last_made_move(self):
        """Returns the last made move, if applicable. If no moves have been made, returns None.
        """
        if self.board.engine.chess_board.move_stack:
            return None
        return self.board.engine.chess_board.peek().uci()

    def is_game_over(self):
        """Returns True if the game is over.
        """
        return self.board.engine.chess_board.is_game_over()

    def reset_board(self):
        '''Skeleton method for resetting board after play.
        '''
        # TODO: implement
        print("Resetting board")

    def send_uci_move_to_board(self, uci_move):
        """Validates uci move, and sends to board if valid.
        """
        if self.board.is_valid_move(uci_move):
            try:
                self.board.make_move(uci_move)
            except NotImplementedError as nie:
                # TODO: update this to do some actual error handling
                raise NotImplementedError(nie.__str__())
        else:
            raise NotImplementedError("Need to handle case of invalid move input. "
                                      "Should we loop until move is valid? What if "
                                      "the board is messed up? Need to revisit.")

    # TODO: this makes implicit assumption that we do human vs. ai. Try to factor that out
    # TODO: convert to class based and store all passed parameters as class members
    def process(self, player):
        '''One iteration of main game loop.

        Note: expects caller to check game.is_game_over before calling.
        There may be moves remaining on self.board.msg_queue after game is over, so it is legal to
            call process after is_game_over returns True as long as board.msg_queue is nonempty.

        Returns:
            made_move: boolean that is True if turn changes, otherwise False.
        '''
        board_status = self.board.get_status_from_arduino()
        print(f"\nBoard Status: {board_status}")

        if board_status.status == ArduinoStatus.EXECUTING_MOVE:
            # Wait for move in progress to finish executing
            time.sleep(1) # reduce the amount of polling while waiting for move to finish

            # TODO: This is just so we have game loop working, remove once we read from arduino
            self.board.set_status_from_arduino(ArduinoStatus.IDLE,
                                               self.board.msg_queue[0].move_count,
                                               None)
            # Turn doesn't change, since we don't get next move if Arduino is still executing
            return False

        if board_status.status == ArduinoStatus.ERROR:
            # TODO: figure out edge/error cases and handle them here
            raise ValueError("Unimplemented, need to handle errors")

        if board_status.status == ArduinoStatus.IDLE:
            if self.board.msg_queue:
                # Arduino sends and receives move_count % 2, for normal move type messages, since
                # it only needs move_count as a form of acknowledgement.
                # Note: debug messages (such as retransmit) are added to message queue as a string.
                # If the move count returned in ArduinoStatus is debug type, just check last char
                # of most recent msg on queue for equality. Otherwise, check that the move count
                # matches move count % 2 (as this is the outgoing format for move count)

                # TODO: Update this to handle adding arbitrary string messages to queue.
                # Probably need to update the Message class to handle being created from a string.

                # if any([
                #     all([board_status.move_count in ArduinoStatus.DEBUG_IDENTIFIERS,
                #          board_status.move_count == self.board.msg_queue[0].move_count]),
                #     board_status.move_count == self.board.msg_queue[0].move_count % 2]):
                if all([board_status.move_count == self.board.msg_queue[0].move_count,
                        board_status.status == ArduinoStatus.IDLE]):
                    self.board.msg_queue.popleft()
                    print("Removed message from queue")
                    # After removing move from queue, return, allows rechecking if msg_queue empty
                    return False

            if self.board.msg_queue:
                self.board.dispatch_msg_from_queue()
                # If moves still in queue, we just try to empty queue, don't get any new move
                return False

            # TODO: verify this works with different modes of interaction
            msg = player.select_move(self.board.engine)

            if msg is None:
                # If msg is None, we have run out of moves in our test file, returning None
                # indicates this is the case.
                return None
            # msg can be a UCI move, or it can be a fully formatted `Message` string
            if len(msg) in (4, 5):
                self.send_uci_move_to_board(msg)
            elif len(msg) == OpCode.MESSAGE_LENGTH:
                # Decompose each move into a `Message` type and add to board's msg queue
                op_code = msg[1]
                if op_code in OpCode.UCI_MOVE_OPCODES:
                    # TODO: need to handle promotions here; maybe also needs to be updated
                    # in general for the opcodes. This section has to be more fleshed out.
                    start_bc = util.BoardCell(ord(msg[2]) - ord('A'), ord(msg[3]) - ord('A'))
                    end_bc = util.BoardCell(ord(msg[4]) - ord('A'), ord(msg[5]) - ord('A'))
                    # If black pieces are facing player, need to transpose board cells.
                    if not self.board.human_plays_white_pieces:
                        start_bc = util.transpose_boardcell(start_bc)
                        end_bc = util.transpose_boardcell(end_bc)
                    # TODO: if human_plays_white_pieces is false, need to flip boardcell diagonally
                    self.board.add_move_to_queue(start_bc, end_bc, op_code)
                    return False
                if op_code == OpCode.INSTRUCTION:
                    # Note: Instruction type opcodes are added to front of queue as they take
                    # priority. Ex: want to request home axis before sending any other msgs.
                    self.board.add_instruction_to_queue(op_type=msg[-1], extra=msg[2])
                    return False
                raise ValueError(f"Couldn't parse message {msg}")

            else:
                raise ValueError(f"Received invalid message {msg}")

            # TODO: After every move, center piece that was just moved on its new square. Need to
            # account for castles as well.

            print(f"{player} made move: {msg}")

            # Status.write_game_status_to_disk(board)

            # If we got here, we added a new move to the move queue, so we flip turn
            return True

        raise ValueError("We shouldn't reach this point in the function.")

if __name__ == '__main__':
    print("No main for this file, please use `cliinterface.py`")
