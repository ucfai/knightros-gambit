import chess
from stockfish import Stockfish
import time
from random import seed
from random import randint
import sys

## path variables and stock fish initialization

mac_stockfish_path = "/usr/local/bin/stockfish"
stockfish_path = mac_stockfish_path

stockfish = Stockfish(stockfish_path)
stockfish.set_elo_rating(1350)

################################################


def fen_to_board(fen): ## function to convert fenn state to 2D array
    board = []
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( ['--'] * int(c) )
            elif c == 'p':
                brow.append( 'bp' )
            elif c == 'P':
                brow.append( 'wp' )
            elif c > 'Z':
                brow.append( 'b'+c.upper() )
            else:
                brow.append( 'w'+c )
        board.append( brow )
    return board

def assign_piece_color():
    seed(1)
    rand = randint(0,2)
    if(rand==0):
        piece_color = "white"
    else:
        piece_color = "black"

    return piece_color

def valid_move(move): ## eventually will actually need to validate move
    return True

def alpha_zero(stockfish,gamestate): ## will eventually be our own AI algorithm, using stockfish just for demonstration
    return stockfish.get_best_move()

def send_move(move): ## will need to take the message and be able to send appropriate serial message to arduino
    time.sleep(1)
    arduino_msg = "msg"
    return True

total_time = 900

print("Welcome to Knightro's Gambit")
print("Type go to start the game")
start = input() 
piece_color = assign_piece_color()
print("Your piece color is " + piece_color)

#Eventually we want to capture the image and pass that to our CV to find the edges

while(1): ## main game loop

    print("Enter your move, time remaining: " + str(total_time) + " seconds") 
    start_time = time.time()    
    move = input() ## eventually will capture image and move will be found that way

    if (valid_move(move)): ## stockfish does not do move validation? 

        ###### This code is simulating the timer #########
        end_time = time.time()
        time_elapsed = end_time - start_time
        total_time  = total_time - time_elapsed
        ##################################################

        stockfish.make_moves_from_current_position([move]) 
        gamestate = fen_to_board(stockfish.get_fen_position())

        print(gamestate)
        ai_move = alpha_zero(stockfish,gamestate) 
    
        print("awaiting arduino.....")

        if(send_move(ai_move)):
            print("move complete ... updating game state")
            stockfish.make_moves_from_current_position([ai_move]) ## after the ai_move
        else:
            print("there was an error when attempting to move the pieces")

    else:
        print("Please enter a valid move and try again")

    
