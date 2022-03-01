import random

from game import Game
import player
from util import parse_test_file, parse_args

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
            return {"mode_of_interaction": "test", "fname": args.test, "players": [player.TestfilePlayer()]}

        # Note: if args.debug specified, takes priority over other modes of operation.
        if args.debug:
            return {"mode_of_interaction": "debug", "players": [player.CLDebugPlayer()]}
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
        if not is_human_turn_at_start:
            players.reverse()

        return {"mode_of_interaction": mode_of_interaction,
                "human_plays_white_pieces": human_plays_white_pieces,
                "players": players}

def add_test_file_messages_to_queue(params):
    '''Add all messages from specified test file to the board move queue.
    '''
    _, fname, board = params
    # Example testfile: 'testfiles/test1.txt'
    # Note: params[1] is filename of test file
    messages, extension = parse_test_file(fname)
    if extension == '.pgn':
        # `messages` is a list of uci_moves
        for uci_move in messages:
            # Decompose each move into a `Message` type and add to board's message queue
            board.send_move_to_board(uci_move)
    elif extension == '.txt':
        for message in messages:
            board.add_message_to_queue(message)

    return board

def get_human_move(mode_of_interaction, board):
    '''Handle human move based on specified mode of interaction.
    '''
    if mode_of_interaction == 'cli':
        return CLHumanPlayer.select_move(board)
    elif mode_of_interaction == 'over_the_board':
        # TODO: think about handling backfill of promotion area if person made a promotion move.
        # If needed, backfill the promotion area (if possible).
        # board.backfill_promotion_area_from_graveyard(color, piece_type)
        pass
    else:
        raise ValueError("Other modes of interaction are unimplemented")

def get_ai_move(ai_player, board):
    '''Handle AI move.
    '''
    return ai_player.select_move(board.engine.fen())

def main():
    '''Main driver loop for running Knightro's Gambit.
    '''
    # Set random seed for program
    random.seed()

    print("Welcome to Knightro's Gambit")

    params = init_parameters()
    mode_of_interaction = params["mode_of_interaction"]
    print(f"\nRUNNING IN {mode_of_interaction.upper()} MODE...\n")

    if params["mode_of_interaction"] == "test":
        game = Game(params["mode_of_interaction"])
        # board = add_test_file_messages_to_queue(params)
        print(board.msg_queue)
        # TODO: Refactor to handle dispatching moves using the code in `process`.
        raise ValueError("Test mode of interaction not yet implemented.")
    elif params["mode_of_interaction"] == "debug":
        game = Game(params["mode_of_interaction"])
        # TODO: Implement debug mode of interaction
        # Should be able to use process with human as both players
        raise ValueError("Debug mode of interaction not yet implemented.")
    elif params["mode_of_interaction"] in ("cli", "otb", "web", "speech"):
        game = Game(params["mode_of_interaction"], params["human_plays_white_pieces"])
        # _, human_plays_white_pieces, board, ai_player = params

    # Main game loop
    current_player = 0
    players = params["players"]
    while True:
        made_move = game.process(players[current_player])

        if made_move:
            # TODO: update this to use circular generator
            current_player = (current_player + 1) % len(players)

        if game.is_game_over():
            if not player_wants_rematch():
                print("Thanks for playing")
                reset_board()
                break  # Break out of main game loop

            print("Ok, resetting board")
            reset_board()

if __name__ == '__main__':
    main()
