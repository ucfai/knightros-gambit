'''Entry point for the Knightr0's Gambit software that controls automatic chessboard.
'''
import time

from boardinterface import Board
from player import CLHumanPlayer, StockfishPlayer
from status import ArduinoStatus

class Game:
    def __init__(self, mode_of_interaction, human_plays_white_pieces=None):    
        self.mode_of_interaction = mode_of_interaction
        # TODO: Set up board with either white or black on human side.
        self.board = Board(human_plays_white_pieces)
        # board.setup_board(is_human_turn)

        # TODO: remove this after real Arduino communication is set up
        self.board.set_status_from_arduino(ArduinoStatus.IDLE, 0, None)

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

    def player_wants_rematch():
        '''Skeleton method for querying player about rematch.
        '''
        # TODO: implement
        return False

    # TODO: this makes implicit assumption that we do human vs. ai. Try to factor that out
    # TODO: convert to class based and store all passed parameters as class members
    def process(self, player):
        '''One iteration of main game loop.

        Returns:
            made_move: boolean that is True if turn changes, otherwise False.
        '''
        # TODO: Handle game end condition here, rematch, termination, etc.
        if self.is_game_over():
            # If game is over, return None for is_human_turn
            return False

        board_status = self.board.get_status_from_arduino()
        print(f"Board Status: {board_status}")

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

            if self.board.msg_queue:
                self.board.dispatch_msg_from_queue()
                # If moves still in queue, we just try to empty queue, don't get any new move
                return False

            self.board.show_on_cli()

            # TODO: verify this works with different modes of interaction
            uci_move = player.select_move(self.board.engine)

            try:
                self.board.send_move_to_board(uci_move)
            except NotImplementedError as nie:
                # TODO: update this to do some actual error handling
                raise NotImplementedError(nie.__str__())

            print(f"{player} made move: {uci_move}")
            # TODO: After every move, center piece that was just moved on its new square. Need to
            # account for castles as well.

            # Status.write_game_status_to_disk(board)

            # If we got here, we added a new move to the move queue, so we flip turn
            return True

if __name__ == '__main__':
    print("No main for this file, please use `cliinterface.py`")
