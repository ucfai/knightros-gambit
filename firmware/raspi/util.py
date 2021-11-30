'''Helper file for miscellaneous utility classes and functions.
'''

class BoardCell:
    '''Helper class for indexing entirety of board.
    '''
    def __init__(self, row=None, col=None):
        '''Point, if no args passed, initialized to (0,0), top left corner of board

        Corresponds to center of board square
        '''
        self.row = row if row else 0
        self.col = col if col else 0

    def __str__(self):
        # TODO: Update this to have a clearer naming scheme for the indexing.
        return f"{chr(self.row+ord('a'))}{chr(self.col+ord('a'))}"

    def get_coords(self):
        '''Returns tuple of row and col indices.
        '''
        return (self.row, self.col)
