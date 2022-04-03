'''CLI entry point for the Knightr0's Gambit software that controls automatic chessboard.
'''
import random

from game import Game
import player
import status
from util import parse_args

# TODO: make this function have a better name; it doesn't just return bool, it also assigns color.
def is_human_turn_at_start():
    '''Assigns piece color for human and returns boolean accordingly.
    '''
    while True:
        start = input("Choose piece color ([w]hite, [b]lack, or [r]andom): ").lower()
        if start == 'r':
            piece_color = "w" if random.randint(0, 1) else "b"
            return piece_color == 'w' # return True if piece color for human is white
        if start == 'b':
            return False
        if start == 'w':
            return True
        print("Please choose one of [w], [b], or [r].")

def init_parameters():
    """Initialize parameters needed to create Game object.

    Return value depends on mode of operation. All return values are dictionaries.
    All return values include:
        mode_of_interaction: Used in initializing Game object.
        players: Main program loop iterates over this array and processes each player in turn.
    If neither of args.test or args.debug are specified, dict also includes:
        human_plays_white_pieces: bool used to specify orientation of board.
    """
    args = parse_args()

    # TODO: update program to handle otb communication and play.
    if args.microcontroller:
        raise ValueError("Serial communication not yet implemented.")

    # TODO: Find better way to initialize board if running in test or debug mode.
    # Also need to update so that we don't assume CLI for setting is_human_turn.
    if args.test or args.debug:
        human_plays_white_pieces = None
        # Note: priority of modes of operation:
        # test > debug > cli == otb == web == speech
        if args.test:
            return {"mode_of_interaction": "test",
                    "players": [player.TestfilePlayer(args.test)],
                    "run_on_hardware": args.microcontroller}

        # Note: if args.debug specified, takes priority over other modes of operation.
        if args.debug:
            return {"mode_of_interaction": "debug", "players": [player.CLDebugPlayer()],
                    "run_on_hardware": args.microcontroller}
    else:
        # Get desired piece color for human. Can be white, black, or random.
        human_plays_white_pieces = is_human_turn_at_start()

        mode_of_interaction = args.playstyle
        if mode_of_interaction == "cli":
            print("Using CLI mode of interaction for human player")
        # TODO: update this to handle physical, web, speech interaction
        else:
            raise ValueError("Other modes of interaction are unimplemented")

        players = [player.CLHumanPlayer(), player.StockfishPlayer(elo_rating=1400)]
        if not human_plays_white_pieces:
            players.reverse()

        return {"mode_of_interaction": mode_of_interaction,
                "players": players,
                "human_plays_white_pieces": human_plays_white_pieces,
                "run_on_hardware": args.microcontroller}

    raise ValueError("Error parsing parameters...")

def player_wants_rematch():
    '''Skeleton method for querying player about rematch.
    '''
    # TODO: implement
    return False

def main():
    '''Main driver loop for running Knightro's Gambit.
    '''
    # Set random seed for program
    random.seed()

    print("Welcome to Knightro's Gambit")

    params = init_parameters()
    mode_of_interaction = params["mode_of_interaction"]
    print(f"\nRUNNING IN {mode_of_interaction.upper()} MODE...\n")

    # Note: If running in test or debug, need to call `alignAxis()` for xmin and ymin explicitly.
    # All other modes perform homing before entering main game loop.
    if params["mode_of_interaction"] == "test":
        game = Game(params["mode_of_interaction"])
        # TODO: Refactor to handle dispatching moves using the code in `process`.
        # raise ValueError("Test mode of interaction not yet implemented.")
    elif params["mode_of_interaction"] == "debug":
        game = Game(params["mode_of_interaction"])
        # TODO: Implement debug mode of interaction
        # Should be able to use process with human as both players
        # raise ValueError("Debug mode of interaction not yet implemented.")
    elif params["mode_of_interaction"] in ("cli", "otb", "web", "speech"):
        game = Game(params["mode_of_interaction"], params["human_plays_white_pieces"])
        # _, human_plays_white_pieces, board, ai_player = params

        # Sends instruction messages for homing (see InstructionType in status)
        game.board.add_instruction_to_queue(op_type=status.InstructionType.ALIGN_AXIS, extra='4')

    # Main game loop
    current_player = 0
    players = params["players"]

    # Show game at start before any moves are made
    game.board.show_on_cli()
    while True:
        # TODO: need to finish processing remainder of moves on the queue after last move
        made_move = game.process(players[current_player])

        if made_move is None:
            print("Finished parsing test file.")
            return

        if made_move:
            # TODO: update this to use circular generator
            current_player = (current_player + 1) % len(players)

            # Show game once after each move is made
            game.board.show_on_cli()

        if game.is_game_over():
            # Need to empty move queue to play out final captures, etc (if any) before ending
            while game.board.msg_queue:
                game.process(None)

            print("\nGAME OVER: ", end="")
            if game.winner() is None:
                print("The game was a draw")
            elif game.winner():
                print("White won")
            else:
                print("Black won")

            if not player_wants_rematch():
                print("Thanks for playing")
                break  # Break out of main game loop

            # TODO: Implement rematch capability
            raise ValueError("Rematch not yet implemented")
            # print("Ok, resetting board")
            # reset_board()

if __name__ == '__main__':
    main()
