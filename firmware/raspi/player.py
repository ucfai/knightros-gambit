"""This file contains classes that wrap players of different types.

This class will wrap the custom AI we build, but for now we use it to get moves from stockfish.

Note: To encapsulate player mechanics in Game class, all players accept a python-chess board as a
    parameter. This is not necessarily used when selecting the move, but is for some of the classes.
"""
# TODO: Uncomment `torch` and `mcts` imports when mcts implementation is finished and integrated.
# import torch

# from mcts import MCTS
from status import OpCode
from util import create_stockfish_wrapper, parse_test_file

class StockfishPlayer:
    """"AI" class that is a simple wrapper around the Stockfish engine.
    """
    def __init__(self, elo_rating=1300):
        self.stockfish = create_stockfish_wrapper()
        self.stockfish.set_elo_rating(elo_rating)

    def select_move(self, board):
        """Sets stockfish state from provided fen and returns best move.
        """
        self.stockfish.set_fen_position(board.fen())
        return self.stockfish.get_best_move()

    def __str__(self):
        return "StockfishPlayer"

# TODO: Uncomment this import when mcts files and model from `chess-AI` have been moved to
# firmware directory.
# class Knightr0Player:
#     """"AI" class that is a wrapper around our custom modification of AlphaZero.
#     """
#     def __init__(self, path_to_model):
#         # TODO: Decide where to store the model, i.e. do we upload to GitHub and have
#         # hard coded path here, or is it best to just pass from game.py. For now, just
#         # pass from game.py
#         self.model = torch.load(path_to_model)
#         # TODO: Update exploration rate.
#         self.mcts = MCTS(exploration_rate=0.01)

#     def select_move(self, board, fen):
#         """Sets stockfish state from provided fen and returns best move.
#         """
#         return mcts.get_best_move(fen, model)

class CLHumanPlayer:
    """"Human" class that allows playing with the chessboard through CLI.
    """
    def __init__(self):
        pass

    def select_move(self, board):
        """Prompts user to select a move.

        Note: board argument included for consistency with `Player` select_move api.
        """
        uci_move = None
        while uci_move is None:
            input_move = input("Please input your move (xyxy): ").lower()
            if board.is_valid_move(input_move):
                uci_move = input_move
            else:
                print(f"The move {input_move} is invalid; please use format (xyxy) e.g., d2d4")
        return uci_move

    def __str__(self):
        return "CLHumanPlayer"

# class PhysicalHumanPlayer:
#     def __init__(self):
#         pass
# TODO: think about handling backfill of promotion area if person made a promotion move.
# If needed, backfill the promotion area (if possible).
# board.backfill_promotion_area_from_graveyard(color, piece_type)

# class WebHumanPlayer:
#     def __init__(self):
#         pass


# class SpeechHumanPlayer:
#     def __init__(self):
#         pass

class CLDebugPlayer:
    """"Human" class that allows playing with the chessboard through CLI."""
    def __init__(self):
        pass

    def select_move(self, board):
        """Prompts user to select a move.
        """
        msg = None
        # uci_move = None
        while msg is None:
            input_move = input("Please enter a uci move (xyxy) or an opcode type message "
                               "(~<OPCODE>xxxx<MOVE_COUNT>): ")
            if len(input_move) in (4, 5) and board.is_valid_move(input_move.lower()):
                msg = input_move.lower()
                break
            if len(input_move) == OpCode.MESSAGE_LENGTH:
                # TODO: Consider doing validation of the inputted opcode message here.
                msg = input_move
                break
            print(f"The move {input_move} is invalid. Please enter a uci move (xyxy) or an "
                  "opcode type message (~<OPCODE>xxxx<MOVE_COUNT>).")
        return msg

    def __str__(self):
        return "CLDebugPlayer"

class TestfilePlayer:
    # TODO: fix docstring
    """"Human" class that allows playing with the chessboard through CLI.
    """
    def __init__(self, fname):
        # Example testfile: 'testfiles/test1.txt'
        self.messages, self.extension = parse_test_file(fname)
        self.current_msg = 0

    def select_move(self, board):
        """Iterates through `messages` array and returns next move.
        """
        # TODO: Update this to use a generator.
        if self.current_msg >= len(self.messages):
            return None

        msg = self.messages[self.current_msg]
        self.current_msg += 1

        return msg

    def __str__(self):
        return "TestfilePlayer"
