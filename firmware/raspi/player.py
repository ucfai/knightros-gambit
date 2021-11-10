class Player:
    def select_move(self, board):
        pass


class StockfishPlayer(Player):
    def __init(self):
        super().__init__()

    def select_move(self, board):
        return board.engine.stockfish.get_best_move()


class CLHumanPlayer(Player):
    def __init(self):
        super().__init__()

    def select_move(self, board):
        legal_moves = board.valid_moves_from_position()
        move = None
        while move is None:
            try_move = input("Please input your move (xyxy): ").lower()
            if try_move in legal_moves:
                move = try_move
            else:
                print(f"The move {try_move} is invalid; please use format (xyxy) e.g., d2d4")
        return move


# TODO: implement
# class PhysicalHumanPlayer(HumanPlayer):
#     def __init(self):
#         super().__init__()


# TODO: implement
# class WebHumanPlayer(HumanPlayer):
#     def __init(self):
#         super().__init__()


# TODO: implement
# class SpeechHumanPlayer(HumanPlayer):
#     def __init(self):
#         super().__init__()

