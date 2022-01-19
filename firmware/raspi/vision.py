'''Helper file to get board state from images of board.
'''
# import cv2 as cv

import util

class BoardStateDetector:
    def __init__(self, calibration_img):
        self.prev_occ_grid = None
        self.board_size = 8
        # self.type_classifier = torch.load(path_to_piece_classifier_model)
        # Note: type_classifier returns 0, 1, 2, 3, 4; convert to 'p', 'q', 'b', 'n', 'r'
        self.type_map = {0: 'p', 1: 'q', 2: 'b', 3: 'n', 4: 'r'}
        # self.col_classifier = torch.load(path_to_piece_classifier_model)
        # Note: col_classifier returns 0, 1, 2; convert to '.', 'w', or 'b' respectively
        self.col_map = {0: '.', 1: 'w', 2: 'b'}
        self.trans_M, self.max_w, self.max_h = self.compute_img_trans_matrix(calibration_img)
        self.img_dim = (800, 800)
        self.sq_size = int(self.img_dim[0] / 8)

    # TODO: Implement detecting image corners, 1) hard code them or 2) compute at startup.
    @staticmethod
    def get_board_corners(board_img):
        pass

    def compute_img_trans_matrix(self, calibration_img):
        return None, None, None
        # TODO: Once we figure out how to detect corners, can uncomment below

        # corners = BoardStateDetector.get_board_corners(calibration_img)

        # Below code taken from pyimagesearch article:
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

        # rect = np.zeros((4, 2), dtype = "float32")
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

    def get_occupancy_grid(self, img_arr_2d):
        '''
        '.': empty, 'w': white piece, 'b': black piece.
        '''
        return [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        # TODO: After classifier is trained/loaded in constructor, can uncomment below.
        # return [[self.col_map[self.col_classifier.predict(img_arr_2d[i, j])]
        #          for i in range(self.board_size)] for j in range(self.board_size)]

    def get_occupancy_diff(self, curr_occ_grid, prev_occ_grid):
        # curr_occ_grid[i][j] is either 'b', 'w', or '.' if empty.
        # Same for prev_occ_grid.
        diff = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                # same color handled implicitly, diff_grid[i][j] already 0
                if (curr_occ_grid[i] == 'b' and prev_occ_grid[i][j] in ('w', '.')) or \
                   (curr_occ_grid[i] == 'w' and prev_occ_grid[i][j] in ('b', '.')):
                    if 1 in diff:
                        diff[1].append((i, j))
                    else:
                        diff[1] = [(i, j)]
                if curr_occ_grid[i] == '.' and prev_occ_grid[i][j] in ('w', 'b'):
                    if -1 in diff:
                        diff[-1].append((i, j))
                    else:
                        diff[-1] = [(i, j)]

        return diff

    @staticmethod
    def get_move_from_diff(diff):
        source, dest = None, None
        # Normal move, includes captures
        if len(diff[-1]) == 1:
            source = diff[-1][0]
            dest = diff[1][0]
        elif len(diff[-1]) == 2:
            # Castle
            if len(diff[1]) == 2:
                if (0, 4) in diff[-1]:
                    # White Kingside
                    if (0, 6) in diff[1]:
                        return 'e1g1'
                    # White Queenside
                    return 'e1c1'
                if (7, 4) in diff[-1]:
                    # Black Kingside
                    if (7, 6) in diff[1]:
                        return 'e8g8'
                    # Black Queenside
                    return 'e8c8'
            # En passant
            else:
                dest = diff[1][0]
                source = [val for val in diff[-1] if val[1] != dest[1]][0]
        else:
            raise RuntimeError("Computer vision detected an invalid change in board state."
                              f"Diff generated between subsequent states = {diff}")

        return util.uci_move_from_boardcells(util.BoardCell(*source), util.BoardCell(*dest))

    def get_piece_type_from_square_img(self, square_img):
        # Use classifier to identify and return piece type at specified square in curr_board_img
        val = 0
        # TODO: After classifier is trained/loaded in constructor, can uncomment below.
        # val = self.type_classifier.predict(square_img)
        return self.type_map[val] if val in self.type_map else None

    # TODO: Implement
    def align_and_segment_image(self, curr_board_img):
        '''Takes in image of board and returns 2d np array of straightened/segmented image.

        First, the corners are detected (or hard coded definition of corners are used), then the
        image is transformed s.t. all four corners of the board are the four corners of the image.
        Then, math is used to splice out each board cell and these are stored in a 2d array where
        each element of the array corresponds to a cell containing at most a single piece.
        '''
        # trans_img = cv.warpPerspective(curr_board_img, self.trans_M, (self.max_w, self.max_h))
        # reduc_trans_img = cv.resize(trans_img, self.img_dim, interpolation = cv.INTER_AREA)
        # img_arr_2d = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        # for i in range(self.board_size):
        #     for j in range(self.board_size):
        #         img_arr_2d[i][j] = reduc_trans_img[i * self.sq_size: (i + 1) * self.sq_size,
        #                                            j * self.sq_size: (j + 1) * self.sq_size,
        #                                            :]
        return curr_board_img

    def get_current_board_state(self, prev_board_fen, curr_board_img):
        img_arr_2d = self.align_and_segment_image(curr_board_img)
        curr_occ_grid = self.get_occupancy_grid(img_arr_2d)
        move = BoardStateDetector.get_move_from_diff(self.get_occupancy_diff(curr_occ_grid,
                                                                             self.prev_occ_grid))
        if util.is_promotion(prev_board_fen, move):
            idx = util.get_chess_coords_from_square(move[2:])
            piece_type = self.get_piece_type_from_square_img(img_arr_2d[idx.row, idx.col])
            if piece_type == 'p' or not piece_type:
                raise RuntimeError("There was a problem detecting the type of the promoted piece. "
                                   "Please use the GUI to indicate which piece you promoted to.")
            move += piece_type
        self.prev_occ_grid = curr_occ_grid
        return move

def main():
    # TODO: Add tests of occupancy grid logic
    pass

if __name__ == '__main__':
    main()
