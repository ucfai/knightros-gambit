"""
adapted code from:
*    Title: chess_tk
*    Author: Guatam Sharma
*    Date: May 19, 2016
*    Availability: https://github.com/saavedra29/chess_tk
"""
import tkinter as tk
from tkinter import simpledialog
import sys
import random

from player import GUIHumanPlayer
import chessboard
from game import Game
import player
from util import parse_args

class GUI:
    """GUI class that creates a visual interface of the board.
    """
    pieces = {}
    selected_piece = None
    focused = None
    turn = ""
    move_first = ""
    game = None
    players = None
    made_move = None
    images = {}
    color1 = "#949494"
    color2 = "#f8f6f1"
    highlightcolor = "#4f4f4f"
    rows = 8
    columns = 8
    dim_square = 64


    def __init__(self, parent, board):
        self.chessboard = board
        self.parent = parent
        canvas_width = self.columns * self.dim_square
        canvas_height = self.rows * self.dim_square
        self.canvas = tk.Canvas(parent, width=canvas_width,
                               height=canvas_height, bg="black")
        self.canvas.pack(padx=8, pady=8)
        #Adding Frame
        self.btmfrm = tk.Frame(parent, height=64)
        self.btmfrm.pack(fill="x", side=tk.BOTTOM)

    def setup_game(self):
        """Sets up the main game variables and then calls the tkinter mainloop
        which waits for events
        """
        params = init_parameters()
        mode_of_interaction = params["mode_of_interaction"]
        print(f"\nRUNNING IN {mode_of_interaction.upper()} MODE...\n")
 
        if params["mode_of_interaction"] == "test":
            self.game = Game(params["mode_of_interaction"], params["interact_w_arduino"])
            # TODO: Refactor to handle dispatching moves using the code in `process`.
            # raise ValueError("Test mode of interaction not yet implemented.")
        elif params["mode_of_interaction"] == "debug":
            self.game = Game(params["mode_of_interaction"], params["interact_w_arduino"])
            # TODO: Implement debug mode of interaction
            # Should be able to use process with human as both players
            # raise ValueError("Debug mode of interaction not yet implemented.")
        elif params["mode_of_interaction"] in ("cli", "otb", "web", "speech"):
            self.game = Game(params["mode_of_interaction"], params["interact_w_arduino"],
                                params["human_plays_white_pieces"])
            # _, human_plays_white_pieces, board, ai_player = params

        self.players = params["players"]
        if params["human_plays_white_pieces"]:
            self.turn = 0
            self.move_first = "gui"
        else:
            self.turn = 1
            self.move_first = "ai"
          
        if not self.turn:
            print("Please make your move on the board.")
        else:
            self.game.process(self.players[0])
            self.chessboard.show(self.game.current_fen())
            self.draw_board()
            self.show_move(self.game.last_made_move(), self.game.is_white_turn())
            self.draw_pieces()
            self.turn -= 1
        self.parent.after(100, self.update_state)
        self.parent.mainloop()

    def update_state(self):
        made_move = self.game.process(self.players[self.turn])
        if made_move:
            self.selected_piece = None
            self.focused = None
            self.pieces = {}
            if self.turn:
                self.chessboard.show(self.game.current_fen())
                self.show_move(self.game.last_made_move(), self.game.is_white_turn())
            self.draw_board()
            self.draw_pieces()
            self.turn = (self.turn + 1) % 2
        self.parent.after(10, self.update_state)

    def square_clicked(self, event):
        """Waits for click on the board and depending on player's turn, executes the move
        """
        if self.turn == 0:
            col_size = row_size = self.dim_square
            selected_column = int(event.x / col_size)
            selected_row = 7 - int(event.y / row_size)
            pos = self.chessboard.alpha_notation((selected_row, selected_column))
            try:
                piece = self.chessboard[pos] #pylint: disable=unused-variable
            except: #pylint: disable=bare-except
                pass
            if self.selected_piece:
                self.made_move = self.shift(self.selected_piece[1], pos)
                if self.made_move == None:
                    self.selected_piece = None
                    print("Invalid Move! Choose a new piece.")
                    return
                self.players[self.turn].set_move(self.made_move)
            self.focus(pos)
            self.draw_board()

        if self.game.is_game_over():
            print("Checking if game is over...")
            # Need to empty move queue to play out final captures, etc (if any) before ending
            while self.game.board.msg_queue:
                self.game.process(None)
            #gui player can no longer make moves
            self.canvas.unbind("<Button-1>")
            print("\nGAME OVER: ", end="")
            if self.game.winner() is None:
                print("The game was a draw")
            elif self.game.winner():
                print("White won")
            else:
                print("Black won")

            if not player_wants_rematch():
                print("Thanks for playing!")
                #break  # Break out of main game loop

            # TODO: Implement rematch capability
            raise ValueError("Rematch not yet implemented")
            # print("Ok, resetting board")
            # reset_board()


    def shift(self, p1, p2):
        """Checks if the move from p1 to p2 is valid, then makes and
        returns the move
        """
        piece = self.chessboard[p1]
        try:
            dest_piece = self.chessboard[p2]
        except: #pylint: disable=bare-except
            dest_piece = None
        if dest_piece is None or dest_piece.color != piece.color:
            try:
                self.chessboard.shift(p1, p2)
            except chessboard.ChessError as error:
                return None
            else:
                return p1.lower()+p2.lower()

    def focus(self, pos):
        """Focuses piece in given position
        """
        try:
            piece = self.chessboard[pos]
        except: #pylint: disable=bare-except
            piece = None
        if piece is not None and (piece.color == self.chessboard.player_turn):
            self.selected_piece = (self.chessboard[pos], pos)
            self.focused = list(map(self.chessboard.num_notation,
                               (self.chessboard[pos].moves_available(pos))))
    def draw_board(self):
        """Prints the board grid.
        """
        self.canvas.create_text(235, 505, fill="#363333",font=("Times", 19,),
            justify=tk.LEFT,
            text="  a            b            c            d"
            "            e            f            g           h  ")
        self.canvas.create_text(505, 265, fill="#363333",font=("Times", 19,),
            text="\n".join("8  7  6  5  4  3  2  1  "))
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x_1 = (col * self.dim_square)
                y_1 = ((7 - row) * self.dim_square)
                x_2 = x_1 + self.dim_square
                y_2 = y_1 + self.dim_square
                if (self.focused is not None and (row, col) in self.focused):
                    self.canvas.create_rectangle(x_1, y_1, x_2, y_2,
                                                 fill=self.highlightcolor,
                                                 tags="area")
                else:
                    self.canvas.create_rectangle(x_1, y_1, x_2, y_2, fill=color,
                                                 tags="area")
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.pieces[name] = (self.pieces[name][0], self.pieces[name][1])
            x_0 = (self.pieces[name][1] * self.dim_square) + int(
                self.dim_square / 2)
            y_0 = ((7 - self.pieces[name][0]) * self.dim_square) + int(
                self.dim_square / 2)
            self.canvas.coords(name, x_0, y_0)
        self.canvas.tag_raise("occupied")
        self.canvas.tag_lower("area")

    def show_move(self, move, color):
        """Prints arrow showing the move from previous to current position.
        """
        self.canvas.after(7000, lambda:self.canvas.delete("arrow"))
        arr = list(move)
        corr_x, corr_y = self.chessboard.num_notation(arr[0].upper() + arr[1])
        corr_x1, corr_y1 = self.chessboard.num_notation(arr[2].upper() + arr[3])
        x_1 =( corr_y * self.dim_square) + int(self.dim_square / 2)
        y_1 =((7 - corr_x) * self.dim_square) + int(self.dim_square / 2)
        x_2 = ( corr_y1 * self.dim_square) + int(self.dim_square / 2)
        if not color:
            y_2 =((7 - corr_x1) * self.dim_square) + int(self.dim_square / 2)+25
        else:
            y_2 =((7 - corr_x1) * self.dim_square) + int(self.dim_square / 2)-25

        self.canvas.create_line(x_1, y_1, x_2, y_2, arrow=tk.LAST, fill="#000000", tags="arrow")

    def draw_pieces(self):
        """Prints the chess pieces.
        """
        self.canvas.delete("occupied")
        for coord, piece in self.chessboard.items():
            corr_x, corr_y = self.chessboard.num_notation(coord)
            if piece is not None:
                filename = f"pieces_image/{piece.shortname.lower()}{piece.color}.png"
                piecename = f"{piece.shortname}{corr_x}{corr_y}"
                if filename not in self.images:
                    self.images[filename] = tk.PhotoImage(file=filename)
                self.canvas.create_image(0, 0, image=self.images[filename],
                                         tags=(piecename, "occupied"),
                                         anchor="c")
                x_0 = (corr_y * self.dim_square) + int(self.dim_square / 2)
                y_0 = ((7 - corr_x) * self.dim_square) + int(self.dim_square / 2)
                self.canvas.coords(piecename, x_0, y_0)

    def gui_loop(self):
        """allows user to select move on board and then calls the event loop """
        self.canvas.bind("<Button-1>", self.square_clicked)
        self.setup_game()


 # TODO: make this function have a better name; it doesn"t just return bool, it also assigns color.
def assign_human_turn_at_start():
     """Assigns piece color for human and returns boolean accordingly.
     """
     while True:
        start = simpledialog.askstring(
            title="Choose Piece Color",
            prompt="Choose piece color ([w]hite, [b]lack, or [r]andom):").lower()
         #start = input("Choose piece color ([w]hite, [b]lack, or [r]andom): ").lower()
        if start == "r":
             piece_color = "w" if random.randint(0, 1) else "b"
        if start == "b":
             return False
        if start == "w":
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
 
    # TODO: need to update so that we don't assume CLI for setting is_human_turn.
    if args.test or args.debug:
        human_plays_white_pieces = None
        # Note: priority of modes of operation:
        # test > debug > cli == otb == web == speech
        if args.test:
            return {"mode_of_interaction": "test",
                "players": [player.TestfilePlayer(args.test)],
                "interact_w_arduino": args.microcontroller}

        # Note: if args.debug specified, takes priority over other modes of operation.
        if args.debug:
            return {"mode_of_interaction": "debug", "players": [player.CLDebugPlayer()],
                "interact_w_arduino": args.microcontroller}
    else:
        # Get desired piece color for human. Can be white, black, or random.
        human_plays_white_pieces = assign_human_turn_at_start()

        mode_of_interaction = args.playstyle
        if mode_of_interaction == "cli":
            print("Using CLI mode of interaction for human player")
        # TODO: update this to handle physical, web, speech interaction
        else:
            raise ValueError("Other modes of interaction are unimplemented")

        players = [player.GUIHumanPlayer(), player.StockfishPlayer(elo_rating=1400)]
        if not human_plays_white_pieces:
            players.reverse()

        return {"mode_of_interaction": mode_of_interaction,
                "players": players,
                "human_plays_white_pieces": human_plays_white_pieces,
                "interact_w_arduino": args.microcontroller}
 
    raise ValueError("Error parsing parameters...")

def player_wants_rematch():
    """Skeleton method for querying player about rematch.
    """
    # TODO: implement
    return False

def main(): #took out chessboard
    """Main driver loop for running Knightro's Gambit.
    """
    # Set random seed for program
    random.seed()
    root=tk.Tk()
    root.title("Knightr0\'s Gambit")

    #initialize gui
    color = None
    gui = GUI(root, chessboard.Board(color))
    textbox=tk.Text(root)
    textbox.pack()

    #redirects print statements to gui
    def redirector(input_str):
        textbox.insert(tk.INSERT, input_str)

    sys.stdout.write = redirector

    print("Welcome to Knightro's Gambit!\n")
    # Show game at start before any moves are made
    gui.draw_board()
    gui.draw_pieces()
    gui.gui_loop()

if __name__ == "__main__":
    main()
