"""Module containing class to display current board state.

Also contains some custom Exceptions.

Based on https://github.com/saavedra29/chess_tk
"""
from copy import deepcopy
import re

import pieces

class Board(dict):
    """Board class to disply current board state.
     """
    y_axis = ("A", "B", "C", "D", "E", "F", "G", "H")
    x_axis = (1, 2, 3, 4, 5, 6, 7, 8)
    captured_pieces = {"white": [], "black": []}
    player_turn = None
    halfmove_clock = 0
    fullmove_number = 1
    history = []
    START_PATTERN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    def __init__(self, pat=None):
        super().__init__()
        self.show(Board.START_PATTERN)

    def is_in_check_after_move(self, p1, p2):
        """Checks if the king is in check after move has been made
        """
        tmp = deepcopy(self)
        tmp.move(p1, p2)
        return tmp.king_in_check(self[p1].color)

    def shift(self, p1, p2):
        """Checks if a move is valid and then makes the move
        """
        p1, p2 = p1.upper(), p2.upper()
        piece = self[p1]
        try:
            dest = self[p2]
        except Exception as e:
            dest = None
        if self.player_turn != piece.color:
            raise NotYourTurn("Not " + piece.color + "'s turn!")
        enemy = ("white" if piece.color == "black" else "black")
        moves_available = piece.moves_available(p1)
        if p2 not in moves_available:
            raise InvalidMove
        if self.all_moves_available(enemy):
            if self.is_in_check_after_move(p1, p2):
                raise Check
        if not moves_available and self.king_in_check(piece.color):
            raise CheckMate
        elif not moves_available:
            raise Draw
        else:
            self.move(p1, p2)
            self.complete_move(piece, dest, p2)

    def move(self, p1, p2):
        """Moves a piece from p1 to p2
        """
        piece = self[p1]
        try:
            _ = self[p2]
        except Exception as e:
            pass
        del self[p1]
        self[p2] = piece

    def complete_move(self, piece, dest, p2):
        """Makes a complete move by updating values and changes the turn to the opponent
        """
        enemy = ("white" if piece.color == "black" else "black")
        if piece.color == "black":
            self.fullmove_number += 1
        self.halfmove_clock += 1
        self.player_turn = enemy
        abbr = piece.shortname
        if abbr == "P":
            abbr = ""
            self.halfmove_clock = 0
        if dest is None:
            movetext = abbr + p2.lower()
        else:
            movetext = abbr + "x" + p2.lower()
            self.halfmove_clock = 0
        self.history.append(movetext)

    def all_moves_available(self, color):
        """Returns an array of all the available moves
        """
        result = []
        for coord in self.keys():
            if (self[coord] is not None) and self[coord].color == color:
                moves = self[coord].moves_available(coord)
                if moves: result += moves
        return result

    def occupied(self, color):
        """Returns an array of occupied coordinates
        """
        result = []
        for coord in iter(self.keys()):
            if self[coord].color == color:
                result.append(coord)
        return result

    def position_of_king(self, color):
        """Returns the position of the king
        """
        for pos in self.keys():
            if isinstance(self[pos], pieces.King) and self[pos].color == color:
                return pos

    def king_in_check(self, color):
        """Checks if the king is in check
        """
        kingpos = self.position_of_king(color)
        opponent = ("black" if color == "white" else "white")
        for _ in self.items():
            return bool(kingpos in self.all_moves_available(opponent))

    def alpha_notation(self, xycoord):
        """Returns x,y coordinates in numbers and letters (e.g., "e6") of
        position on chessboard.
        """
        if xycoord[0] < 0 or xycoord[0] > 7 or xycoord[1] < 0 or xycoord[
            1] > 7:
            return
        return self.y_axis[int(xycoord[1])] + str(self.x_axis[int(xycoord[0])])

    def num_notation(self, coord):
        """Returns x,y coordinates in numbers of position on chessboard.
        """
        return int(coord[1]) - 1, self.y_axis.index(coord[0])

    def is_on_board(self, coord):
        """Checks if the given coord is on the board
        """
        list = [coord[1] >= 0, coord[1] <= 7, coord[0] >= 0, coord[0] <= 7]
        return all(list)

    def show(self, pat):
        """Prints a visual representation of the board state based on a
        fen string.
        """
        self.clear()
        pat = pat.split(" ")

        def expand(match):
            return " " * int(match.group(0))

        pat[0] = re.compile(r"\d").sub(expand, pat[0])
        for x, row in enumerate(pat[0].split("/")):
            for y, letter in enumerate(row):
                if letter == " ":
                    continue
                coord = self.alpha_notation((7 - x, y))
                self[coord] = pieces.create_piece(letter)
                self[coord].place(self)
        if pat[1] == "w":
            self.player_turn = "white"
        else:
            self.player_turn = "black"
            self.halfmove_clock = int(pat[4])
            self.fullmove_number = int(pat[5])

class ChessError(Exception):
    pass

class Check(ChessError):
    pass

class InvalidMove(ChessError):
    pass

class CheckMate(ChessError):
    pass

class Draw(ChessError):
    pass

class NotYourTurn(ChessError):
    pass

class InvalidCoord(ChessError):
    pass