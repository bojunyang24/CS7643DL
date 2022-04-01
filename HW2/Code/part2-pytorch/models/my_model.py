import torch
import torch.nn as nn
from torchvision import models

# You will re-use the contents of this file for your eval-ai submission.

class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        
        self.sequential_net = nn.Sequential(
            nn.Conv2d(3, 64*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64*2, 128*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128*2),

            nn.Conv2d(128*2, 256*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(256*2, 256*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),
            nn.BatchNorm2d(256*2),

            nn.Conv2d(256*2, 256*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(256*2, 256*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),
            nn.BatchNorm2d(256*2),

            nn.Conv2d(256*2, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),
            nn.BatchNorm2d(512),

            nn.Flatten(),
            nn.Linear(512*2*2, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 10)
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        outs = self.sequential_net(x)

        # outs = self.resnet(x)
        # outs = torch.flatten(outs, start_dim=1)
        # outs = self.resnetfc1(outs)
        # outs = self.r1(outs)
        # outs = self.resnetfc2(outs)
        # outs = self.r2(outs)
        # outs = self.resnetfc3(outs)


        # outs = self.conv1(x)
        # outs = self.bn1(outs)
        # outs = self.r1(outs)
        # outs = self.conv2(outs)
        # outs = self.r2(outs)
        # outs = self.pool(outs)
        # outs = self.conv3(outs)
        # outs = self.r3(outs)

        # outs = torch.flatten(outs, start_dim=1)
        # outs = self.fc1(outs)
        # outs = self.r1(outs)
        # outs = self.fc2(outs)
        # outs = self.r2(outs)
        # outs = self.fc3(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs