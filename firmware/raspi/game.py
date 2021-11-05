import chess
from stockfish import Stockfish


def fen_to_board(fen):
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

mac_stockfish_path = "/usr/local/bin/stockfish"
stockfish_path = mac_stockfish_path

stockfish = Stockfish(stockfish_path)
stockfish.set_elo_rating(1350)
fenpos1 = stockfish.get_fen_position()
print(fen_to_board(fenpos1)) ## prints array of current position.
