'''This file prototypes the conversion from the network output representation to a chess move.

Uses https://ai.stackexchange.com/questions/27336/how-does-the-alpha-zeros-move-encoding-work?rq=1
for reference.
'''
import numpy as np

class convertToMove:

    def convert_output_to_gan_move(self,square, move):

        '''Construct move xyxy from the given parameters.

        Need to convert information about square and move to GAN (e.g. e1e4)
        We need to know whether white or black turn (so that we know relative N S etc)
        This is so we know moving 2 spaces N is either increasing 2 or decreasing 2
        '''

        files = ['a','b','c','d','e','f','g','h']
        initial_file = files[square[0]]
        initial_rank = square[1] 
        initial_gan = initial_file + str(initial_rank)
    
        start_coordinates = np.array([square[0],square[1]])

        if(move[0] == "knight" or move[0] == "underpromotion"):
            value_move =  1
        else:
            value_move = move[0]

        move_directions = {
            "N": np.array([0,value_move]),
            "S": np.array([0,-value_move]),
            "E": np.array([value_move,0]),
            "W": np.array([-value_move,0]),
            "NE":np.array([value_move,value_move]),
            "NW":np.array([-value_move,value_move]),
            "SW":np.array([-value_move,-value_move]),
            "SE":np.array([value_move,-value_move]),
        }

        if(move[0] == "knight"):
            end_coordinates = start_coordinates + (2 * move_directions[move[1]]) + move_directions[move[2]]
            end_gan = files[end_coordinates[0]] + str(end_coordinates[1])
        elif(move[0] == "underpromotion"):
            end_coordinates = start_coordinates + move_directions[move[1]]
            end_gan = files[end_coordinates[0]] + str(end_coordinates[1]) + move[2]
        else:
            end_coordinates = start_coordinates + move_directions[move[1]]
            end_gan = files[end_coordinates[0]] + str(end_coordinates[1])

        gan = initial_gan + end_gan
       
        return gan

    def find_best_move(self,policy):
        '''Entry point for driver demonstrating conversion from policy to move.
        '''
        # We create 56 entries for each `(n_squares, direction)` pair
        codes, i = {}, 0
        # Can move min of 1 square, max of 7 squares
        for n_squares in range(1, 8):
            # 8 directions
            for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                codes[(n_squares,direction)] = i
                i += 1

        assert len(codes) == 56

        # The knight moves we'll encode as the long "two"-cell edge move first
        # and the short "one"-cell edge second:
        for two in ["N","S"]:
            for one in ["E","W"]:
                codes[("knight", two, one)] , i = i , i + 1
        for two in ["E","W"]:
            for one in ["N","S"]:
                codes[("knight", two, one)] , i = i , i + 1

        # Now we should have 64 codes. As I understand, the final 9 moves are
        # when a pawn reaches the final rank and chosen to be underpromoted.
        # It can reach teh final rank either by moving N, or by capturing NE,
        # NW. Underpromotion is possible to three pieces. Writing the code:

        for move in ["N","NW","NE"]:
            for promote_to in ["R","K","B"]:
                codes[("underpromotion", move, promote_to)] , i = i , i + 1

        # The above gives us 73 codes
        assert len(codes) == 73

        # for code in codes:
        #   print(code)

        # Create a flattened vector of 4672 moves, reshape to have size 8, 8, 73
        #policy = np.arange(8 * 8 * 73)
        #policy = policy[::-1]

        # TODO: we need to account for invalid moves, to do later
        # This gets three indices (x, y, z) of maximum value in policy 3d array
        imax = np.unravel_index(policy.argmax(), policy.shape)

        square = (imax[0], imax[1])
        move = list(codes)[imax[2]]

        print(f"Ideal move is to do move {move} from square {square}")
        ret = self.convert_output_to_gan_move(square, move)
        return ret

