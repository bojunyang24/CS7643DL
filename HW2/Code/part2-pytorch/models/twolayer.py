import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        # c, h, w = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.a1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        x = torch.flatten(x, start_dim=1)

        out = self.fc1(x)
        out = self.a1(out)
        out = self.fc2(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out