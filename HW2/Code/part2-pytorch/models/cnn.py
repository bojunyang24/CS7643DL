import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv_params = {'k': 7, 's': 1, 'p': 0}
        self.pool_params = {'k': 2, 's': 2}
        out_channels = 32
        out_features = 10
        cifar10_in_channels = 3*32*32
        self.conv1 = nn.Conv2d(
            3,
            out_channels,
            self.conv_params['k'],
            self.conv_params['s'],
            self.conv_params['p']
        )
        self.a1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            self.pool_params['k'],
            self.pool_params['s']
        )
        h = 32
        conv_out_h = int(((32 + (2 * self.conv_params['p']) - (self.conv_params['k'] - 1) - 1) / self.conv_params['s']) + 1)
        pool_out_h = int(((conv_out_h - (self.pool_params['k'] - 1) - 1) / self.pool_params['s']) + 1)
        fc_in = out_channels * pool_out_h * pool_out_h
        self.fc1 = nn.Linear(fc_in, out_features)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv1(x)
        outs = self.a1(outs)
        outs = self.pool(outs)
        outs = torch.flatten(outs, start_dim=1)
        outs = self.fc1(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs