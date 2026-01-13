import torch
import torch.nn as nn
import torch.nn.functional as F

class RoverCNN(nn.Module):
    def __init__(self, output_dim):
        super(RoverCNN, self).__init__()

        # Canal 0: Rover Position.
        # Canal 1: Obstacles.
        # Canal 2: Objectives.
        # Canal 3: Visit History.
        # Input: (Batch, 4, 8, 8)

        # Convolution 1: Analyzes the immediate surroundings.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        # Convolution 2: Analysis of more complex patterns.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.flatten_dim = 64 * 8 * 8

        self.fc_common = nn.Linear(self.flatten_dim, 512) # Common layer before separation.
        self.fc_value = nn.Linear(512, 1) # Stream 1: The Value (V) of the state (1 output only).
        self.fc_advantage = nn.Linear(512, output_dim) # Stream 2: The Advantage (A) of each action (4 outputs).

    def forward(self, x):
        # Vision.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        # Common.
        x = F.relu(self.fc_common(x))

        # Separation
        val = self.fc_value(x)          # V(s).
        adv = self.fc_advantage(x)      # A(s, a).

        return val + (adv - adv.mean(dim=1, keepdim=True))