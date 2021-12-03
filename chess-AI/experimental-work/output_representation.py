'''This file prototypes the conversion from the network output representation to a chess move.

Uses https://ai.stackexchange.com/questions/27336/how-does-the-alpha-zeros-move-encoding-work?rq=1
for reference.
'''
import numpy as np
import chess

class PlayNetworkPolicyConverter:
    '''Class for converting output of PlayNetwork policy to a UCI chess move.
    '''
    def __init__(self):
        # We create 56 entries for each `(n_squares, direction)` pair
        self.codes_dict, i = {}, 0
        # Can move min of 1 square, max of 7 squares
        for n_squares in range(1, 8):
            # 8 directions
            for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                self.codes_dict[(n_squares,direction)] = i
                i += 1

        assert len(self.codes_dict) == 56

        # The knight moves we'll encode as the long "two"-cell edge move first
        # and the short "one"-cell edge second:
        for two in ["N","S"]:
            for one in ["E","W"]:
                self.codes_dict[("knight", two, one)] , i = i , i + 1
        for two in ["E","W"]:
            for one in ["N","S"]:
                self.codes_dict[("knight", two, one)] , i = i , i + 1

        # Now we should have 64 codes. As I understand, the final 9 moves are
        # when a pawn reaches the final rank and chosen to be underpromoted.
        # It can reach teh final rank either by moving N, or by capturing NE,
        # NW. Underpromotion is possible to three pieces. Writing the code:

        for move in ["N","NW","NE"]:
            for promote_to in ["r","n","b"]:
                self.codes_dict[("underpromotion", move, promote_to)] , i = i , i + 1

        # The above gives us 73 codes
        assert len(self.codes_dict) == 73

        self.codes_list = list(self.codes_dict)
        self.files_from_idx = {
            0: 'a',
            1: 'b',
            2: 'c',
            3: 'd',
            4: 'e',
            5: 'f',
            6: 'g',
            7: 'h'
        }
        self.idx_from_file = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7
        }

    def convert_policy_indices_to_uci_move(self, indices, board_t):
        '''Construct move in UCI format from indices corresponding to neural network output cell.

        Returns None if the move is not a legal chess move.
        Need to convert information about square and move to UCI (e.g. e1e4)
        We need to know whether white or black turn (so that we know relative N S etc)
        This is so we know moving 2 spaces N is either increasing 2 or decreasing 2

        Note: this validates move w/in context of current game state, returns None if invalid move.
        '''
        square = (indices[0], indices[1])
        move = self.codes_list[indices[2]]

        start_sq_str = self.files_from_idx[square[0]] + str(square[1])
        start_coords = np.array(square)

        # Set number of squares to move selected piece
        if move[0] == "knight" or move[0] == "underpromotion":
            n_squares =  1
        else:
            n_squares = move[0]

        # Set directionality of move according to whose turn it is to play
        # Network output is always w.r.t. perspective of player whose turn it is
        if board_t.turn == chess.BLACK:
            n_squares = -n_squares

        move_directions = {
            "N": np.array([0, n_squares]),
            "S": np.array([0, -n_squares]),
            "E": np.array([n_squares, 0]),
            "W": np.array([-n_squares, 0]),
            "NE":np.array([n_squares, n_squares]),
            "NW":np.array([-n_squares, n_squares]),
            "SW":np.array([-n_squares, -n_squares]),
            "SE":np.array([n_squares, -n_squares]),
        }

        if move[0] == "knight":
            # Here, len(move_directions[move[1]]) == 1, same for move[2]
            end_coords = start_coords + np.array([2 * move_directions[move[1]],
                                                  move_directions[move[2]]])
        else:
            # Here, len(move_directions[move[1]]) == 2
            end_coords = start_coords + move_directions[move[1]]

        # Make sure all indices are in bounds
        if end_coords[0] > 7 or end_coords[0] < 0 or start_coords[0] > 7 or start_coords[0] < 0:
            return None

        # Construct end square string
        end_sq_str = self.files_from_idx[end_coords[0]] + str(end_coords[1])
        # If needed, append piece type to which to promote
        if move[0] == "underpromotion":
            end_sq_str += move[2]
        # If not underpromotion, check if move is queen promotion
        elif chess.Move.from_uci(end_sq_str + 'q') in board_t.legal_moves:
            end_sq_str += 'q'

        uci_move = start_sq_str + end_sq_str

        # Validate move, return only if legal
        return uci_move if chess.Move.from_uci(uci_move) in board_t.legal_moves else None

    def convert_uci_move_to_policy_indices(self, uci_move, board_t):
        '''Convert a uci move to a tuple of three indices corresponding to a cell in the policy.
        '''
        start_sq = uci_move[:2]
        start_coords = [self.idx_from_file[uci_move[0]], int(uci_move[1])]
        end_coords = [self.idx_from_file[uci_move[2]], int(uci_move[3])]

        # If black to play, compute coordinates for flipped board
        if board_t.turn == chess.BLACK:
            start_coords[0], start_coords[1] = 7 - start_coords[0], 7 - start_coords[1]
            end_coords[0], end_coords[1] = 7 - end_coords[0], 7 - end_coords[1]

        # Value of movement in E/W direction
        ew_value = end_coords[0] - start_coords[0]
        # Value of movement in N/S direction
        ns_value = end_coords[1] - start_coords[1]

        if board_t.piece_type_at(chess.parse_square(start_sq)) == chess.KNIGHT:
            # Need to get which of 8 possible knight moves: NE, NW, SE, SW, EN, ES, WN, WS
            # (2, 1): "NE",
            # (2, -1): "NW",
            # (-2, 1): "SE",
            # (-2, -1): "SW",
            # (2, 1): "EN",
            # (2, -1): "ES",
            # (-2, 1): "WN",
            # (-2, -1): "WS",
            if ew_value in (2, -2):
                direction = "E" if ew_value > 0 else "W"
                direction += "N" if ns_value > 1 else "S"
            if ns_value in (2, -2):
                direction = "N" if ns_value > 0 else "S"
                direction += "E" if ew_value > 0 else "W"
            move_idx = self.codes_list.index(("knight", direction[0], direction[1]))
            return np.array([*start_coords, move_idx])

        if len(uci_move) == 5:
            # handle promotions/underpromotions
            if uci_move[4] in ['b', 'r', 'n']:
                direction = "N"
                if ew_value != 0:
                    direction += "E" if ew_value > 0 else "W"
            else:
                print(f"queen promotion: {uci_move[4]}")
            move_idx = self.codes_list.index(("underpromotion", direction, uci_move[4]))
            return np.array([*start_coords, move_idx])

        # If we are here, normal move along single file/rank/diagonal
        assert ew_value == 0 or ns_value == 0 or ew_value == ns_value

        if ew_value == 0:
            direction = "N" if ns_value > 0 else "S"
        else:
            if ns_value == 0:
                direction = ""
            else:
                direction = "N" if ns_value > 0 else "S"
            direction += "E" if ew_value > 0 else "W"

        return np.array([*start_coords, self.codes_list.index((max(ew_value, ns_value),
                                                               direction))])

    def find_best_move(self, policy, board_t):
        '''Find the best move according to the policy outputted by the network.
        '''
        # This gets three indices (x, y, z) of maximum value in policy 3d array
        max_indices = np.unravel_index(policy.argmax(), policy.shape)
        return self.convert_policy_indices_to_uci_move(max_indices, board_t)

    # TODO: consider consolidating `find_best_legal_move` and `find_value_of_all_legal_moves`
    def find_best_legal_move(self, policy, board_t):
        '''Return the legal move with the highest value as given by the policy.
        '''
        max_val_move = None
        max_val = -float('inf')

        for move in board_t.legal_moves:
            uci_move = move.uci()
            indices = self.convert_uci_move_to_policy_indices(uci_move, board_t)
            val = policy[indices]
            if val > max_val:
                max_val_move = uci_move
                max_val = val

        return max_val_move

    def find_value_of_all_legal_moves(self, policy, board_t):
        '''Returns a dictionary containing the value of each legal move in the current board state.
        '''
        move_vals = {}
        for move in board_t.legal_moves:
            uci_move = move.uci()
            i, j, k = self.convert_uci_move_to_policy_indices(uci_move, board_t)
            move_vals[uci_move] = policy[i, j, k]

        return move_vals

def main():
    '''Entry point for driver to test output representation converter.
    '''
    # Create a flattened vector of 4672 moves, reshape to have size 8, 8, 73
    policy = np.arange(8 * 8 * 73).reshape((8, 8, 73))

    # Create a python-chess board
    board = chess.Board()
    policy_converter = PlayNetworkPolicyConverter()

    move_values = policy_converter.find_value_of_all_legal_moves(policy, board)
    for move, value in move_values.items():
        print(f"{move}: {value}")

    # TODO: make test cases here, one for each type of move to verify that output
    # is being parsed correctly both for uci -> indices and indices -> uci
    # Functions should be inverse of each other.

if __name__ == "__main__":
    main()
