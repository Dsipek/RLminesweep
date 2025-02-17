import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONV_UNITS, DENSE_UNITS

class CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(input_shape[0], CONV_UNITS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CONV_UNITS, CONV_UNITS, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(CONV_UNITS, CONV_UNITS, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(CONV_UNITS, CONV_UNITS, kernel_size=3, padding=1)

        self.conv_output_size = self._get_conv_output_size(input_shape)

        self.fc1 = nn.Linear(self.conv_output_size, DENSE_UNITS)
        self.fc2 = nn.Linear(DENSE_UNITS, DENSE_UNITS)
        self.out = nn.Linear(DENSE_UNITS, n_actions)

    def _get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(1, -1).size(1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Ensure channel dimension exists
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)