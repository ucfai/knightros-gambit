import random
import time

from boardinterface import Board
from player import CLHumanPlayer, StockfishPlayer
from status import ArduinoStatus, Status

def assign_piece_color():
    '''
    Returns either 'w' or 'b' with equal probability
    '''
    return "w" if random.randint(0, 1) else "b"

def is_human_turn_at_start():
    while True:
        start = input("Choose piece color ([w]hite, [b]lack, or [r]andom): ").lower()
        if start == 'r':
            piece_color = assign_piece_color()
            return piece_color == 'w' # return True if piece color for human is white
        elif start == 'b':
            return False
        elif start == 'w':
            return True
        else:
            print("Please choose one of [w], [b], or [r].")

def main():
    random.seed()

    # board initialization
    elo_rating = 1300
    board = Board('OSX', elo_rating)
    print("Welcome to Knightro's Gambit")

    # TODO: update this to handle physical, web, speech interaction
    mode_of_interaction = 'cli'

    # Get desired piece color for human. Can be white, black, or random.
    is_human_turn = is_human_turn_at_start()

    if mode_of_interaction == "cli":
        human_player = CLHumanPlayer()
    else:
        raise ValueError("Other modes of interaction are unimplemented")
    # TODO: update this to handle physical interaction
    # TODO: update this to handle web interaction
    # TODO: update this to handle speech interaction

    ai_player = StockfishPlayer()

    # main game loop
    while True:
        board_status = board.get_status_from_arduino()
        print(board_status)
        # At program start, Arduino is waiting first instruction;
        # need to have some information passed about who moves first
        # TODO: this section of code should only 
        if board_status == ArduinoStatus.WaitingInitialization:
            # Set arduino status (i.e., who moves first, AI or human)
            if is_human_turn:
                board.set_status_from_arduino(ArduinoStatus.WaitingForPersonMove)
            else:
                board.set_status_from_arduino(ArduinoStatus.WaitingForAIMove)
            continue
        elif board_status == ArduinoStatus.ExecutingMove:
            # Wait for move in progress to finish executing
            time.sleep(1) # reduce the amount of polling while waiting for move to finish
            # TODO: This section of code is just here for developing game loop; remove when arduino comms work
            is_human_turn = not is_human_turn
            if is_human_turn:
                board.set_status_from_arduino(ArduinoStatus.WaitingForPersonMove)
            else:
                board.set_status_from_arduino(ArduinoStatus.WaitingForAIMove)
            # END REMOVE
            continue


        board.show_on_cli()

        if board.get_status_from_arduino() == ArduinoStatus.WaitingForPersonMove:
            if mode_of_interaction == 'cli':
                move = human_player.select_move(board)
                if board.is_valid_move(move):
                    board.make_move(move)
                    # TODO: This is just so we have game loop working, remove once we read from arduino
                    board.set_status_from_arduino(ArduinoStatus.ExecutingMove)
                else:
                    # TODO: do error handling
                    raise NotImplementedError("Need to handle case of invalid move input. Should we loop until "
                                              "move is valid? What if the board is messed up? Need to revisit")
            else:
                raise ValueError("Other modes of interaction are unimplemented")
        elif board.get_status_from_arduino() == ArduinoStatus.WaitingForAIMove:
            move = ai_player.select_move(board)
            if board.is_valid_move(move):
                board.make_move(move)
                print(f"AI made move: {move}")
                # TODO: This is just so we have game loop working, remove once we read from arduino
                board.set_status_from_arduino(ArduinoStatus.ExecutingMove)
            else:
                # TODO: do error handling
                raise NotImplementedError("Need to handle case of invalid move input. Should we loop until "
                                          "move is valid? What if the board is messed up? Need to revisit")
        elif board.get_status_from_arduino() == ArduinoStatus.GenericError:
            raise ValueError("Need to handle errors")
        else:
            raise ValueError("We should not end up here")

        Status.write_game_status_to_disk(board)

if __name__ == '__main__':
    main()
