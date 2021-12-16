'''Entry point for the Knightr0's Gambit software that controls automatic chessboard.
'''

import random
import time

from boardinterface import Board
from player import CLHumanPlayer, StockfishPlayer
from status import ArduinoStatus

def assign_piece_color():
    '''
    Returns either 'w' or 'b' with equal probability
    '''
    return "w" if random.randint(0, 1) else "b"

# TODO: make this function have a better name; it doesn't just return bool, it also assigns color.
def is_human_turn_at_start():
    '''Assigns piece color for human and returns boolean accordingly.
    '''
    while True:
        start = input("Choose piece color ([w]hite, [b]lack, or [r]andom): ").lower()
        if start == 'r':
            piece_color = assign_piece_color()
            return piece_color == 'w' # return True if piece color for human is white
        if start == 'b':
            return False
        if start == 'w':
            return True
        print("Please choose one of [w], [b], or [r].")

def handle_human_move(mode_of_interaction, board):
    '''Handle human move based on specified mode of interaction.
    '''
    if mode_of_interaction == 'cli':
        uci_move = CLHumanPlayer.select_move(board)
        try:
            board.send_move_to_board(uci_move)
        except NotImplementedError as nie:
            print(nie.__str__())
    elif mode_of_interaction == 'over_the_board':
        # TODO: think about handling backfill of promotion area if person made a promotion move.
        # If needed, backfill the promotion area (if possible).
        # board.backfill_promotion_area_from_graveyard(color, piece_type)
        pass
    else:
        raise ValueError("Other modes of interaction are unimplemented")

def handle_ai_move(ai_player, board):
    '''Handle AI move.
    '''
    uci_move = ai_player.select_move(board.engine.fen())
    try:
        board.send_move_to_board(uci_move)
        print(f"AI made move: {uci_move}")
    except NotImplementedError as nie:
        print(nie.__str__())

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

def main():
    '''Main driver loop for running Knightro's Gambit.
    '''
    random.seed()

    print("Welcome to Knightro's Gambit")

    # TODO: update this to handle physical, web, speech interaction
    mode_of_interaction = 'cli'

    # Get desired piece color for human. Can be white, black, or random.
    is_human_turn = is_human_turn_at_start()

    # TODO: Set up board with either white or black on human side.
    board = Board(human_plays_white_pieces=is_human_turn)
    # board.setup_board(is_human_turn)

    # TODO: remove this after real Arduino communication is set up
    board.set_status_from_arduino(ArduinoStatus.IDLE, 0, None)

    if mode_of_interaction == "cli":
        print("Using CLI mode of interaction for human player")
    else:
        raise ValueError("Other modes of interaction are unimplemented")
    # TODO: update this to handle physical, web, speech interaction

    ai_player = StockfishPlayer(elo_rating=1400)

    # Main game loop
    while True:
        # TODO: Handle game end condition here, rematch, termination, etc.
        if board.engine.is_game_over():
            if not player_wants_rematch():
                print("Thanks for playing")
                reset_board()
                break  # Break out of main game loop
            
            print("Ok, resetting board")
            reset_board()

        board_status = board.get_status_from_arduino()
        print(f"Board Status: {board_status}")

        if board_status.status in (ArduinoStatus.EXECUTING_MOVE,
                                   ArduinoStatus.MESSAGE_IN_PROGRESS):
            # Wait for move in progress to finish executing
            time.sleep(1) # reduce the amount of polling while waiting for move to finish

            # TODO: This is just so we have game loop working, remove once we read from arduino
            board.set_status_from_arduino(ArduinoStatus.IDLE,
                                          board.move_queue[0].move_count % 256,
                                          None)
            continue

        if board_status.status == ArduinoStatus.ERROR:
            # TODO: figure out edge/error cases and handle them here
            raise ValueError("Unimplemented, need to handle errors")

        if board.move_queue:
            # Arduino sends and receives move_count % 256, since it can only transmit one byte
            if all([board_status.move_count == board.move_queue[0].move_count % 256,
                    board_status.status == ArduinoStatus.IDLE]):
                board.move_queue.popleft()

        if board.move_queue:
            board.dispatch_move_from_queue()
            continue

        board.show_on_cli()

        if is_human_turn:
            handle_human_move(mode_of_interaction, board)
        else:
            handle_ai_move(ai_player, board)
        # TODO: After every move, center piece that was just moved on its new square
        # TODO: Need to account for castles as well.

        is_human_turn = not is_human_turn
        # Status.write_game_status_to_disk(board)

if __name__ == '__main__':
    main()
