''' This script generates labeled data for developing classifiers for piece color and piece type.

Note: This script requires the CLI tool `pgn-extract` be installed.

The script does the following for each image/move in the user-specified `game_name` folder:
    1) Create a subdirectory `game_name/j`, where `j` represents the current move (0-indexed).
    2) Use the PGN to generate a fen string for move `j` and save it to `game_name/j/board.fen`.
    3) Use fen to create 2D grid of board state (where each cell consists of color and piece type
    if it is occupied, else '.').
    4) Align and segment (into 64 images, one per board square) the image corresponding to move j.
    5) Save each square as a single image with path `game_name/j/square.png` where `square` is the
    cell on the chessboard for that piece (e.g. `e4`).
    6) Generate two labels for each `square`, saving these in `game_name/j/labels.csv` (i.e.
    each row is of the form `square, color_label, type_label`)
        a) COLOR: 0, 1, 2; these correspond to '.', 'w', 'b') and are used for developing square
        occupancy detection using color.
        b) TYPE: 0, 1, 2, 3, 4, 5; these correspond to 'empty', 'p', 'q', 'b', 'n', 'r' and are
        used for developing piece classification capability.

This script assumes all images are taken with `a1` corresponding to the bottom left corner of
the board in the image.
'''

from pathlib import Path
import os
import subprocess

import cv2 as cv

def get_board_corners(calibration_img):
    # TODO: Update this to actually return corners
    return [0, 0, 0, 0]

def compute_img_trans_matrix(calibration_img):
    return None, None, None
    # TODO: uncomment below when we actually get corners
    # corners = get_board_corners(calibration_img)

    # # Below code taken from pyimagesearch article:
    # # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # rect = np.zeros((4, 2), dtype="float32")
    # s = corners.sum(axis=1)
    # rect[0] = corners[np.argmin(s)]
    # rect[2] = corners[np.argmax(s)]
    # diff = np.diff(corners, axis=1)
    # rect[1] = corners[np.argmin(diff)]
    # rect[3] = corners[np.argmax(diff)]
    # (tl, tr, br, bl) = rect

    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))
    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))

    # dst = np.array([
    #     [0, 0],
    #     [maxWidth - 1, 0],
    #     [maxWidth - 1, maxHeight - 1],
    #     [0, maxHeight - 1]], dtype="float32")

    # # compute the perspective transform matrix and then apply it
    # M = cv.getPerspectiveTransform(rect, dst)

    # return M, maxWidth, maxHeight

def align_img(curr_board_img, trans_mat, max_w, max_h, img_dim):
    '''Image is transformed s.t. all four corners of the board are the four corners of the image.
    '''
    trans_img = cv.warpPerspective(curr_board_img, trans_mat, (max_w, max_h))
    return cv.resize(trans_img, img_dim, interpolation=cv.INTER_AREA)

def segment_image(aligned_img, board_size, sq_size):
    '''Takes in aligned image of board and returns 2d np array of segmented image.

    Math is used to splice out each board cell and these are stored in a 2d array where each
    element of the array corresponds to a cell containing at most a single piece.
    '''
    img_arr_2d = [[None for _ in range(board_size)] for _ in range(board_size)]
    for i in range(board_size):
        for j in range(board_size):
            img_arr_2d[i][j] = aligned_img[i * sq_size: (i + 1) * sq_size,
                                           j * sq_size: (j + 1) * sq_size,
                                           :]
    return curr_board_img

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

def save_img_and_labels(move_path, row, col, sq_img, color_label, type_label):
    '''Expects `move_path` to be of form f"{game_path}/{move_number}".
    '''
    # TODO: should moves be saves as `a1`, etc?
    square = chr(col + ord('a')) + chr(row + ord('1'))
    cv.imwrite(f"{move_path}/square.png", sq_img)
    with open(f"{move_path}/labels.csv", "a") as file:
        file.write(f"{square}, {color_label}, {type_label}")

def get_fens(game_path):
    result = subprocess.run(['pgn-extract', "--quiet", "-Wepd", game_path],
                            stdout=subprocess.PIPE, check=True)
    return [fen for fen in result.stdout.decode('utf-8').split('\n') if len(fen) != 0]

def main():
    '''Main code to iterate over images from a specified game.

    Creates a directory for each move in the game and produces labeled cells for each move to be
    to develop piece color classifier and piece type classifier.
    '''
    # Prompt user for game name until they provide a dir name that already exists.
    game_name = None
    while not game_name:
        game_name = input("Please name this game: ")
        path = Path(os.getcwd() + '/' + game_name)
        if not path.exists():
            print("Please use a directory name that already exists.")
            game_name = None

    # Build list of fens using pgn-extract
    fens =  get_fens(game_path=f"{path.__str__()}/{game_name}.pgn")

    image_paths = list(path.glob("*.png"))

    if not fens or not image_paths:
        return

    # Compute/set metadata for aligning and segmenting image up front based on first board image
    test_img = cv.imread(image_paths[0])
    trans_mat, max_w, max_h = compute_img_trans_matrix(test_img)
    board_size = 8
    img_dim = (800, 800)
    sq_size = img_dim[0] / 8

    for j, (image_path, fen) in enumerate(zip(image_paths, fens)):
        # Create directory for move j
        try:
            # TODO: see about a better way of doing this
            os.mkdir(Path(path.__str__() + f'/{j}').__str__())
        except:
            raise ValueError("Error with creating save dir")

        print(os.path.basename(image_path))
        curr_board_img = cv.imread(image_path)

        # Write fen to `board.fen`
        with open(f'{path.__str__()}/board.fen', mode='w') as file:
            file.write(fen)

        aligned_img = align_img(curr_board_img, trans_mat, max_w, max_h, img_dim)
        img_arr_2d = segment_image(aligned_img, board_size, sq_size)
        board_2d = get_2d_grid(fen)

        move_path = f"{path.__str__()}/{j}"

        # Set up output .csv file
        with open(f"{move_path}/labels.csv", "a") as file:
            file.write(f"{square}, {color_label}, {type_label}")

        for row in range(8):
            for col in range(8):
                color_label, type_label = get_labels(board_2d[row][col])
                save_img_and_labels(move_path, row, col, img_arr_2d[row][col], color_label,
                                                                               type_label)

if __name__ == '__main__':
    main()
