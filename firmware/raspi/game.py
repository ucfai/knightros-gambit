import random
import time
import platform

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
    board = Board(platform.system(), elo_rating)
    print("Welcome to Knightro's Gambit")

    # TODO: update this to handle physical, web, speech interaction
    mode_of_interaction = 'cli'

    # Get desired piece color for human. Can be white, black, or random.
    is_human_turn = is_human_turn_at_start()

    # Set up board with either white or black on human side.
    board.setup_board(is_human_turn)
    # TODO: remove this after real Arduino communication is set up
    board.set_status_from_arduino(ArduinoStatus.Idle)

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
        print(f"Board Status: {board_status}")
        
        if board_status == ArduinoStatus.ExecutingMove or board_status == ArduinoStatus.MessageInProgress:
            # Wait for move in progress to finish executing
            time.sleep(1) # reduce the amount of polling while waiting for move to finish

            # TODO: This is just so we have game loop working, remove once we read from arduino
            board.set_status_from_arduino(ArduinoStatus.Idle)
            continue

        if board_status == ArduinoStatus.Error:
            # TODO: figure out edge/error cases and handle them here
            raise ValueError("Unimplemented, need to handle errors")

        board.show_on_cli()

        if is_human_turn:
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
        else:
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

        # TODO: After every move, center piece that was just moved on its new square
        # TODO: Need to account for castles as well.
        
        is_human_turn = not is_human_turn
        Status.write_game_status_to_disk(board)

if __name__ == '__main__':
    main()
