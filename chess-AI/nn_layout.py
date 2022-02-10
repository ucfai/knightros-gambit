"""Basic layout for AI CNN architecture.

Reference: Alpha Zero Paper pg. 13-14
Input convolutional layer takes (MT + L) channels of input, each channel with N * N dimensions.
N = 8   the dimensions of the chess board,
M = 14  the 12 channels of the position of each piece type and 2 channels for repetition,
T = 8   the number of past moves to consider,
L = 7   the number of constant number planes besides repetition like castling, move count, etc.
"""

import torch
from torch import nn

from output_representation import policy_converter
from state_representation import get_cnn_input


class PlayNetwork(nn.Module):
    """Chess AI CNN with policy vector and value output.

    Input current and 7 past moves, each move state with their own set of channels to represent
    their state, then pass to output. Output has a policy vector size of 4672 possbile moves and
    a value head of one scalar evaluation number.
    """

    def __init__(self):
        super(PlayNetwork, self).__init__()
        self.num_res_blocks = 2
        self.num_filters = 32

        # Takes 119 channels of input comprised of 7 previous states' piece and repetition
        # planes plus the current state input of 21 channels.
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=19,
                                                  out_channels=self.num_filters,
                                                  kernel_size=3,
                                                  padding=1,
                                                  bias=False),
                                        nn.BatchNorm2d(self.num_filters),
                                        nn.ReLU())

        # Each conv layer in a residual block uses 256 3x3 filters, padding used
        # to keep the channel dimensions constant.
        self.res_blocks = [nn.Sequential(nn.Conv2d(in_channels=self.num_filters,
                                                   out_channels=self.num_filters,
                                                   kernel_size=3,
                                                   padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(self.num_filters),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=self.num_filters,
                                                   out_channels=self.num_filters,
                                                   kernel_size=3,
                                                   padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(self.num_filters)) for _ in range(self.num_res_blocks)]

        # Use 2 1x1 filters to convolve input channels to 2 output channels, one
        # representing piece to move, and the other representing move to take out 
        # of possible moves. Use them to pick move from possible moves, represented
        # by 73 channels of 8x8, or 4672 possible moves
        self.policy_head = nn.Sequential(nn.Conv2d(in_channels=self.num_filters,
                                                   out_channels=2,
                                                   kernel_size=1,
                                                   bias=False),
                                         nn.BatchNorm2d(2),
                                         nn.ReLU(),
                                         nn.Flatten(),
                                         nn.Linear(in_features=2 * 8 * 8,
                                                   out_features=8 * 8 * 73))

        # Convolve 256 8x8 channels into 8x8 channel, then use fully connected layer to
        # take 64 input features from 8x8 channel and transform to 256 output features,
        # then transform to one scalar value.
        self.value_head = nn.Sequential(nn.Conv2d(in_channels=self.num_filters,
                                                  out_channels=1,
                                                  kernel_size=1,
                                                  bias=False),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(in_features=8 * 8,
                                                  out_features=self.num_filters),
                                        nn.ReLU(),
                                        nn.Linear(in_features=self.num_filters,
                                                  out_features=1),
                                        nn.Tanh())

    def forward(self, x):
        x = self.conv_layer(x)

        # Go through all the residual blocks and add the
        # input of the block to the output before regularization.
        for res_block in self.res_blocks:
            shortcut = x
            x = res_block(x)
            x = nn.functional.relu_(x + shortcut)

        value_out = self.value_head(x)
        policy_out = self.policy_head(x)

        policy = policy_out.reshape(8, 8, 73)
        return policy, value_out


def main():
    model = PlayNetwork()

    import chess
    board = chess.Board()

    policy, value = model(get_cnn_input(board))
    print(policy)
    print(value)

    move_values = policy_converter.find_value_of_all_legal_moves(policy, board)
    for move, value in move_values.items():
        print(f"{move}: {value}")


if __name__ == "__main__":
    main()
