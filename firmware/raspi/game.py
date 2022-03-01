'''Entry point for the Knightr0's Gambit software that controls automatic chessboard.
'''
import time

from boardinterface import Board
from player import CLHumanPlayer, StockfishPlayer
from status import ArduinoStatus, OpCode

class Game:
    def __init__(self, mode_of_interaction, human_plays_white_pieces=None):    
        self.mode_of_interaction = mode_of_interaction
        # TODO: Set up board with either white or black on human side.
        self.board = Board(human_plays_white_pieces)
        # board.setup_board(is_human_turn)

        # TODO: remove this after real Arduino communication is set up
        self.board.set_status_from_arduino(ArduinoStatus.IDLE, 0, None)

    def winner(self):
        return self.board.engine.outcome()

    def current_fen(self):
        return self.board.engine.fen()

    def is_white_turn(self):
        '''Return True if it is white's turn, False otherwise.
        '''
        return self.board.engine.is_white_turn()

    def last_made_move(self):
        if len(self.board.engine.move_stack):
            return None
        return self.board.engine.peek().uci()

    def is_game_over(self):
        return self.board.engine.is_game_over()

    def reset_board():
        '''Skeleton method for resetting board after play.
        '''
        # TODO: implement
        print("Resetting board")

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
                                               self.board.msg_queue[0].move_count % 10,
                                               None)
            # Turn doesn't change, since we don't get next move if Arduino is still executing
            return False

        if board_status.status == ArduinoStatus.ERROR:
            # TODO: figure out edge/error cases and handle them here
            raise ValueError("Unimplemented, need to handle errors")

        if board_status.status == ArduinoStatus.IDLE:
            if self.board.msg_queue:
                # Arduino sends and receives move_count % 10, since it can only transmit one char for
                # move count
                if all([board_status.move_count == self.board.msg_queue[0].move_count % 10,
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
            message = player.select_move(self.board.engine)

            if message is None:
                # If message is None, we have run out of moves in our test file, returning None
                # indicates this is the case.
                return None
            # message can be a UCI move, or it can be a fully formatted `Message` string
            if len(message) in (4, 5):
                if self.board.is_valid_move(message):
                    try:
                        self.board.make_move(message)
                    except NotImplementedError as nie:
                        # TODO: update this to do some actual error handling
                        raise NotImplementedError(nie.__str__())
                else:
                    raise NotImplementedError("Need to handle case of invalid move input. "
                                              "Should we loop until move is valid? What if "
                                              "the board is messed up? Need to revisit.")
            elif len(message) == OpCode.MESSAGE_LENGTH:
                # Decompose each move into a `Message` type and add to board's message queue
                self.board.add_message_to_queue(message)
            else:
                raise ValueError(f"Received invalid message {message}")

            # TODO: After every move, center piece that was just moved on its new square. Need to
            # account for castles as well.

            print(f"{player} made move: {message}")

            # Status.write_game_status_to_disk(board)

            # If we got here, we added a new move to the move queue, so we flip turn
            return True

if __name__ == '__main__':
    print("No main for this file, please use `cliinterface.py`")
