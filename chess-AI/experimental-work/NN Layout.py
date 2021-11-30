

# this is the basic AI CNN layout
# we will use this and improve it until it can be used for the final product.


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision


class PlayNetwork(nn.Module):
    def __init__(self):
        super(PlayNetwork, self).__init__()
        
        # Basic layout for AI CNN, still need to polish the value head and BatchNorm2d input sizes.
        
        # Reference: Alpha Zero Paper pg. 13-14
        # Input conv layer takes (MT + L) channels of input, each channel with N * N dimensions.
        # N = 8 representing the dimensions of the chess board,
        # M = 14 representing the 12 channels of the position of each piece type and 2 channels for repetition,
        # T = 8 representing the number of past moves to consider,
        # L = 7 representing the number of constant number planes besides repetition such as castling, move count, etc.
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=19, out_channels=256, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))
        
        # Each conv layer in a residual block uses 256 3x3 filters, padding used to keep the channel dimensions constant.
        self.resBlock = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(256), 
                                    nn.ReLU(inplace=True), 
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(256))
        
        # Use 2 1x1 filters to convolve input channels to 2 output channels, one representing piece
        # to move, and the other representing move to take out from possible moves.
        # Use them to pick move from possible moves, represented by 73 channels of 8x8, or 4672 possible moves <- (Revising)
        self.policyHead = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
                                       nn.BatchNorm2d(2),
                                       nn.ReLU(inplace=True),
                                       nn.Flatten(),
                                       nn.Linear(in_features=2*8*8, out_features=73))
        
        # Convolve 256 8x8 channels into 8x8 channel, then use fully connected layer to take 64 input features from 8x8
        # channel and transform to 256 output features, then transform to one scalar value.
        self.valueHead = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU(inplace=True),
                                       nn.Flatten(),
                                       nn.Linear(in_features=8*8, out_features=256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_features=256, out_features=1),
                                       nn.Tanh())

        

    def forward(self, x):
        x = self.convLayer(x)
        
        # Go through all the residual blocks and add the input of the block to the output before regularization.
        for numRes in range(3):
            shortcut = x
            x = self.resBlock(x)
            x = F.relu_(x + shortcut)
        
        value_out = self.valueHead(x)
        policy_out = self.policyHead(x)
        
        return (policy_out, value_out)
    
    
    
model = PlayNetwork()
policy, value = model(torch.randn(1, 19, 8, 8))
policy = policy.reshape(73)
value = value.item()
print(policy)
print(value)



