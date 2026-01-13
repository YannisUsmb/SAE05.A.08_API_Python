import torch
import torch.nn as nn
import torch.nn.functional as F

class RoverCNN(nn.Module):
    def __init__(self, output_dim):
        super(RoverCNN, self).__init__()

        # Canal 1: Rover Position.
        # Canal 2: Obstacles.
        # Canal 3: Base.
        # Input: (Batch, 3, 8, 8)

        # Convolution 1: Analyzes the immediate surroundings.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Convolution 2: Analysis of more complex patterns.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.flatten_dim = 64 * 8 * 8

        # Linear layers for decision-making.
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)