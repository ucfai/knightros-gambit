''' This script generates labeled data for developing classifiers for piece color and piece type.

The script does the following for each image/move in the user-specified <game_name> folder:
    1) Create a subdirectory `<game_name>/j`.
    2) Use the PGN to generate a fen string for move j and save it to `game_name/j/board.fen`.
    3) Use fen to create 2D grid of board state (where each cell consists of color and piece type
    if it is occupied, else None).
    4) Align and segment (into 64 images, one per board square) the image corresponding to move j.
    5) Save each square as a single image with path `<game_name>/j/<square>.png` where `square` is the
    cell on the chessboard for that piece (e.g. `e4`).
    6) Generate two labels for each `square`, saving these in `<game_name>/j/labels.csv`
        a) COLOR: 0, 1, 2; these correspond to '.', 'w', 'b') and are used for developing square
        occupancy detection using color.
        b) TYPE: 0, 1, 2, 3, 4, 5; these correspond to 'empty', 'p', 'q', 'b', 'n', 'r' and are
        used for developing piece classification capability.

This script assumes all images are taken with `a1` corresponding to the bottom left corner of
the board in the image.
'''

from pathlib import Path
import os

def sqs_from_full_board_image(image):
    pass

def get_2d_grid(fen):
    '''Returns a 2d board from fen representation

    Taken from this SO answer:
    https://stackoverflow.com/questions/66451525/how-to-convert-fen-id-onto-a-chess-board
    '''
    board = []
    for row in reversed(fen.split('/')):
        brow = []
        for char in row:
            if char == ' ':
                break
            if char in '12345678':
                brow.extend(['.'] * int(char))
            else:
                brow.append(char)
        board.append(brow)

    return board

def get_labels(piece_w_color):
    '''Returns tuple of color and piece type from provided square.
    '''
    # COLOR: 0, 1, 2; corresponds to '.', 'w', 'b'
    # TYPE: 0, 1, 2, 3, 4, 5; corresponds to 'empty', 'p', 'q', 'b', 'n', 'r'
    if piece_w_color == '.':
        return (0, 0)
    color = 1 if piece_w_color.isupper() else 2

    # Don't need to include empty, as it is handled above.
    type_map = {'p': 1, 'q': 2, 'b': 3, 'n': 4, 'r': 5}
    
    return (color, type_map[piece_w_color.lower()])

def save_img_and_labels(i, j, sq_img, color_label, type_label):
    pass

def main():
    # Prompt user for game name until they provide a dir name that already exists.
    game_name = None
    while not game_name:
        # TODO: Find a better way to check if path exists
        game_name = input("Please name this game: ")
        path = Path(os.getcwd() + '/' + game_name)
        if not path.exists():
            print("Please use a directory name that already exists.")
            game_name = None

    # TODO: Build list of fens using pgn-extract
    fens = []

    image_paths = path.glob("*.png")
    for j, image_path in enumerate(image_paths):
        # TODO: Find a better way to print image name
        print(image_path.__str__().split('/')[-1])

        # Create directory for move j
        save_dir = Path(path.__str__() + f'/{j}')
        try:
            os.mkdir(save_dir.__str__())
        except:
            raise ValueError("Error with creating save dir")
            
        fen = fens[j]
        # TODO: Write fen to `board.fen`

        # TODO: Load image as np array
        image = None
        # TODO: Use `sqs_from_full_board_image` code from cv
        segmented_2d_arr = sqs_from_full_board_image(image)
        board_2d = get_2d_grid(fen)

        for i in range(8):
            for j in range(8):
                color_label, type_label = get_labels(board_2d[i][j])
                # TODO: implement save_image_and_labels
                save_img_and_labels(i, j, segmented_2d_arr[i][j], color_label, type_label)

if __name__ == '__main__':
    main()
