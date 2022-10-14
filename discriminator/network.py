"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim, device, dropout=0.1):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN, self).__init__()
        
        self.device = device

        self.layer1 = nn.Linear(in_dim, 1024)
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.layer3 = nn.Linear(256, 32)
        self.dropout3 = nn.Dropout(dropout)
        self.layer4 = nn.Linear(32, out_dim)

    def forward(self, state):
        """
            Runs a forward pass on the neural network.

            Parameters:
                state - state to pass as input

            Return:
                output - the output of our forward pass
        """
        activation1 = F.relu(self.dropout1(self.layer1(state)))
        activation2 = F.relu(self.dropout2(self.layer2(activation1)))
        activation3 = F.relu(self.dropout3(self.layer3(activation2)))
        output = self.layer4(activation3)

        return output
