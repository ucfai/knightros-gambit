'''Helper file to get board state from images of board.
'''
from chess import Board

from util import BoardCell, uci_move_from_boardcells

class BoardStateDetector:
    def __init__(self):
        self.prev_state = None
        self.prev_occupancy_grid = None
        self.board_size = 8

    def get_occupancy_grid(self, board_state_image):
        '''
        '.': empty, 'w': white piece, 'b': black piece.
        '''
        diff_grid = [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        # TODO: implement logic to classify each cell in image
        return diff_grid

    def get_move_from_occupancy_diff(self, curr_occupancy_grid, prev_occupancy_grid):
        # curr_occupancy_grid[i][j] is either 'b', 'w', or '.' if empty.
        # Same for prev_occupancy_grid.

        diff = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                # same color handled implicitly, diff_grid[i][j] already 0
                if (curr_occupancy_grid[i] == 'b' and prev_occupancy_grid[i][j] in ('w', '.')) or \
                   (curr_occupancy_grid[i] == 'w' and prev_occupancy_grid[i][j] in ('b', '.')):
                    if 1 in diff:
                        diff[1].append((i, j))
                    else:
                        diff[1] = [(i, j)]
                if curr_occupancy_grid[i] == '.' and prev_occupancy_grid[i][j] in ('w', 'b'):
                    if -1 in diff:
                        diff[-1].append((i, j))
                    else:
                        diff[-1] = [(i, j)]

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

        return uci_move_from_boardcells(BoardCell(*source), BoardCell(*dest))

    # TODO: Implement
    def is_promotion(self, prev_board_fen, move):
        prev_board_2d = get_2d_grid(prev_board_fen)

        # if piece in prev_board_img at square move[:2] is a pawn and move[2:] is the final rank, this is a promotion
        pass

    # TODO: Implement
    def get_piece_type(self, curr_board_img, square):
        # Use classifier to identify and return piece type at specified square in curr_board_img
        pass

    def get_current_board_state(self, curr_board_img):
        curr_occupancy_grid = self.get_occupancy_grid(curr_board_img)
        move = self.get_move_from_occupancy_diff(curr_occupancy_grid, self.prev_occupancy_grid)
        if is_promotion(prev_board_img, move):
            move += get_piece_type(curr_board_img, move)
        self.prev_occupancy_grid = curr_occupancy_grid
        self.prev_board_img = curr_board_img
        return move

def main():
    # TODO: Add tests of occupancy grid logic
    pass

if __name__ == '__main__':
    main()
